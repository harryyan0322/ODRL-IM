import gym, json, os
import platform
import resource
import time
from pathlib import Path
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import RandomSampler
# from torch_geometric.data import DataLoader

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from garage.experiment.deterministic import set_seed

from myEnv import MyEnv

TF_ENABLE_ONEDNN_OPTS = 0

from Model import GCN, GNNQFunction, NodeMLP, PaperActor, PaperCritic

from atac.atac import ATAC
from atac.garage_tools.rl_utils import train_agent, get_sampler, setup_gpu, get_algo, get_log_dir_name, load_algo

from RelayBuffer import (
    Replay_buffer,
    HeterogeneousReplayBuffer,
    getNormalization,
    MyGraphDataset,
)

import graph_utils
from offline_data import dataset_statistics, load_json, load_offline_dataset, load_pickle_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs"

# 训练的主体函数，需要进行大量修改
def train_func(ctxt=None,
               data_path1=None,
               data_config=None,
               community_data_config=None,
               # data_path1='./Wiki-2.txt-b5-n400-new.pkl',
               data_path2=None,
               # path to the data_info
               graph_path=None,  # path to the graph edge list
               # graph_path='./Wiki-2.txt',  # path to the part_info

               # Evaluation mode
               evaluation_mode=False,
               # Trainer parameters
               n_epochs=200,  # number of training epochs
               batch_size=0,  # number of samples collected per update
               # Algorithm parameters
               discount=0,
               policy_lr=5e-6,  # optimization stepsize for policy update
               value_lr=5e-4,  # optimization stepsize for value regression
               alpha_lr=5e-5,  # optimization stepsize for entropy temperature
               target_update_tau=5e-3,  # for target network
               minibatch_size=8,  # optimization/replaybuffer minibatch size
               n_grad_steps=1000,  # number of gradient updates per epoch
               n_warmstart_steps=0,  # number of warm-up steps
               fixed_alpha=None,  # whether to fix the temperate parameter
               use_deterministic_evaluation=True,  # do evaluation based on the deterministic policy
               num_evaluation_episodes=5,  # number of episodes to evaluate (only affect off-policy algorithms)
               reward_scale=1.0,  # scale rewards before constructing Bellman update, only used in off-policy algorithms
               # ATAC parameters
               beta=1.0,  # weight on the Bellman error , 1.0 for ATAC, 0.0 for BC
               norm_constraint=100,
               use_two_qfs=True,  # whether to use two q function
               q_eval_mode='0.5_0.5',
               init_pess=False,
               # Compute parameters
               seed=0,
               n_workers=1,  # number of workers for data collection
               gpu_id=0,  # try to use gpu, if implemented
               force_cpu_data_collection=True,  # use cpu for data collection.
               # Logging parameters
               save_mode='full',
               ignore_shutdown=True,  # do not shutdown workers after training
               return_mode='average',  # 'full', 'average', 'last'
               return_attr='Evaluation/AverageReturn',  # the log attribute
               use_policy_lr_scheduler=False,
               # use_critic_lr_scheduler=False,
               use_std_loss=False,
               data_saving_dir=None,
               use_resource_reward=True,
               use_weighted_adj=True,
               use_MLP=False,
               log_path=None,
               budget=None,
               dataset=None,
               inference_selection_mode='partitioned',
               inference_num_partitions=None,
               inference_partition_seed=42,
               algorithm_mode='atac',
               cql_alpha=1.0,
               gnn_backbone='gcn',
               gnn_depth=3,
               gat_heads=1,
               paper_architecture=False,
               ):
    """ Train an agent in batch mode. """

    # Set the random seed
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize gym env
    # env = DetectionEnv(data_path=data_path0, num_node=200, input_dim=5, num_clique=5)
    if not graph_path:
        raise ValueError("graph_path is required; pass it through a manifest or --graph-path")
    graph = graph_utils.read_graph(graph_path, ind=0, directed=False)
    num_node = graph.num_nodes

    print("num_node==", num_node)
    # print("num_edges==", graph.num_edges)

    # Initialize replay buffer and Load data
    # 加载数据
    community_num_node = None
    if community_data_config:
        with open(community_data_config, "r", encoding="utf-8") as file:
            community_config = json.load(file)
        data_buffer = HeterogeneousReplayBuffer(
            community_config, seed=seed
        )
        transition = data_buffer.sample_transitions_option(1)
        input_dim = np.asarray(transition["obs"][0]).shape[-1]
        if input_dim != 4:
            raise ValueError("Formal PDF protocol requires four node features")
        env_num_node = int(np.asarray(transition["obs"][0]).shape[0])
        # Heterogeneous replay samples come from local Leiden graphs.  The
        # garage environment is only used for its spaces, so its shape and
        # budget must describe the sampled local graph rather than the full
        # graph used later for global inference.
        community_num_node = env_num_node
        vmax = max(
            float(np.max(entry["store"]["rewards"]))
            for entry in data_buffer.entries
        )
        vmin = min(
            float(np.min(entry["store"]["rewards"]))
            for entry in data_buffer.entries
        )
    elif data_config:
        data_list1 = load_offline_dataset(data_config)
        print("offline_data_manifest==", data_list1["manifest"])
        if data_saving_dir:
            report = dataset_statistics(data_list1, budget_percent=float(budget))
            with open(os.path.join(data_saving_dir, "offline_data_stats.json"), "w") as file:
                json.dump(report, file, indent=2, sort_keys=True)
                file.write("\n")
            manifest_config, _ = load_json(data_config)
            with open(os.path.join(data_saving_dir, "offline_data_manifest.json"), "w") as file:
                json.dump(manifest_config, file, indent=2, sort_keys=True)
                file.write("\n")
    else:
        if not data_path1:
            raise ValueError("data_path1 is required when no offline manifest is supplied")
        data_list1 = load_pickle_dataset(data_path1)
    if community_data_config:
        data_list1 = None
    else:
        print(data_list1.keys())
    if data_saving_dir:
        inference_config = {
            "selection_mode": inference_selection_mode,
            "partition_backend": "leiden_rb_configuration",
            "num_partitions": "discovered_from_graph",
            "partition_seed": inference_partition_seed,
            "resolution_parameter": 1.0,
            "directed_partition_projection": "undirected_topology",
        }
        with open(os.path.join(data_saving_dir, "inference_config.json"), "w") as file:
            json.dump(inference_config, file, indent=2, sort_keys=True)
            file.write("\n")
    # print("obs_list=",(data_list1['action_list'][0]))
    #
    if not community_data_config:
        vmax = max(data_list1['rewards'])
        vmin = min(data_list1['rewards'])
    print("vmax==", vmax)
    print("vmin==", vmin)

    if not community_data_config:
        data_buffer = Replay_buffer()
        data_buffer.add(data_list1)
        transition = data_buffer.sample_transitions_option(2)


    # input_dim = transition['obs'][0][0].shape[1]
    # input_dim = transition['obs'][0].shape[1]
        input_dim = np.asarray(transition['obs'][0]).shape[-1]
        if input_dim != 4:
            raise ValueError("Formal PDF protocol requires four node features")
        env_num_node = num_node


    budget_percent = float(budget)
    budget_nodes = max(1, int(
        budget_percent * (community_num_node or env_num_node) / 100.0
    ))
    env = MyEnv(graph, budget_nodes, env_num_node, input_dim)

    # Initialize the algorithm
    # env_spec = env.spec
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')
    print("training_device==", device)

    if not use_MLP and paper_architecture:
        policy = PaperActor(input_dim=input_dim, gat_hidden=64,
                            actor_hidden=128, gat_depth=3, gat_heads=4)
        qf1 = PaperCritic(input_dim=input_dim + 1, hidden_dim=128)
        qf2 = PaperCritic(input_dim=input_dim + 1, hidden_dim=128)
    elif not use_MLP:
        # Actor 
        # policy = GCN(input_dim=obs.shape[-1], output_dim=1)
        policy = GCN(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=1,
            depth=gnn_depth,
            backbone=gnn_backbone,
            gat_heads=gat_heads,
        )

        # 输入为obs和action，输出为Q值
        qf1 = GNNQFunction(input_dim=1, hidden_dim=64, output_dim=1,
                           num_nodes=env_num_node, depth=gnn_depth,
                           backbone=gnn_backbone, gat_heads=gat_heads)
        qf2 = GNNQFunction(input_dim=1, hidden_dim=64, output_dim=1,
                           num_nodes=env_num_node, depth=gnn_depth,
                           backbone=gnn_backbone, gat_heads=gat_heads)
    else:
        policy = NodeMLP(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=1,
            depth=gnn_depth,
        )
        qf1 = NodeMLP(
            input_dim=1,
            hidden_dim=64,
            output_dim=1,
            depth=gnn_depth,
        )
        qf2 = NodeMLP(
            input_dim=1,
            hidden_dim=64,
            output_dim=1,
            depth=gnn_depth,
        )

    sampler = data_buffer.sample_transitions_option

    algo_config = dict(  # union of all algorithm configs
        env_spec=env,
        device=device,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        sampler=sampler,
        replay_buffer=data_buffer,
        discount=discount,
        policy_lr=policy_lr,
        qf_lr=value_lr,
        alpha_lr=alpha_lr,
        target_update_tau=target_update_tau,
        buffer_batch_size=minibatch_size,
        gradient_steps_per_itr=n_grad_steps,
        use_deterministic_evaluation=use_deterministic_evaluation,
        min_buffer_size=int(0),
        num_evaluation_episodes=num_evaluation_episodes,
        fixed_alpha=fixed_alpha,
        reward_scale=reward_scale,
        target_entropy=-1000,  # -np.prod(env_spec.action_space.shape),后续env写完后可以不在此处指定
        data_saving_dir=data_saving_dir,
        Vmin=vmin,
        Vmax=vmax,

    )

    # ATAC
    extra_algo_config = dict(
        beta=beta,
        norm_constraint=norm_constraint,
        use_two_qfs=use_two_qfs,
        n_warmstart_steps=n_warmstart_steps,
        q_eval_mode=q_eval_mode,
        graph=graph,
        use_std_loss=use_std_loss,
        use_lr_scheduler=use_policy_lr_scheduler,
        algorithm_mode=algorithm_mode,
        cql_alpha=cql_alpha,
        # use_critic_lr_scheduler=use_critic_lr_scheduler,

    )

    algo_config.update(extra_algo_config)  # updata()函数把字典参数 dict2 的 key/value(键/值) 对更新到字典 dict 里

    algo = ATAC(**algo_config)  # 初始化了一个ATAC算法实例
    algo.to(device)

    # Initialize the trainer
    from atac.garage_tools.trainer import BatchTrainer as Trainer
    trainer = Trainer(ctxt)
    trainer.setup(algo=algo,
                  env=env,
                  force_cpu_data_collection=force_cpu_data_collection,
                  save_mode=save_mode,
                  return_mode=return_mode,
                  return_attr=return_attr)

    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)


def run(log_root=str(DEFAULT_RUN_ROOT),
        torch_n_threads=2,
        snapshot_frequency=0,
        **train_kwargs):
    torch.set_num_threads(torch_n_threads)  # 设置CPU上用于操作内并行的线程数。
    data_config = train_kwargs.get('data_config')
    community_data_config = train_kwargs.get('community_data_config')
    if data_config:
        manifest, manifest_path = load_json(data_config)
        train_kwargs['data_config'] = str(manifest_path)
        if manifest.get('graph_path'):
            graph_path = os.path.expandvars(os.path.expanduser(manifest['graph_path']))
            if not os.path.isabs(graph_path):
                graph_path = os.path.join(manifest_path.parent, graph_path)
            train_kwargs['graph_path'] = os.path.abspath(graph_path)
        if manifest.get('dataset'):
            train_kwargs['dataset'] = str(manifest['dataset'])
        if manifest.get('budget_percent') is not None:
            train_kwargs['budget'] = f"{float(manifest['budget_percent']):g}"
    elif community_data_config:
        community_data_config = str(Path(community_data_config).resolve())
        manifest = json.loads(Path(community_data_config).read_text(encoding="utf-8"))
        train_kwargs['community_data_config'] = community_data_config
        train_kwargs['graph_path'] = manifest['full_graph_path']
        train_kwargs['dataset'] = manifest['dataset']
        train_kwargs['budget'] = f"{float(manifest['budget_percent']):g}"

    log_dir = get_log_dir_name(train_kwargs, ['beta', 'discount', 'norm_constraint',
                                              'policy_lr', 'value_lr', 'alpha_lr',
                                              'use_two_qfs',
                                              'fixed_alpha',
                                              'q_eval_mode',
                                              'n_warmstart_steps', 'seed', 'trial',
                                              'data_path1','data_config','graph_path','log_path','budget','dataset',
                                              'inference_selection_mode','inference_num_partitions',
                                              'inference_partition_seed','algorithm_mode',
                                              'cql_alpha','gnn_backbone','gnn_depth',
                                              'gat_heads'])
    
    # data_path = train_kwargs['data_path']
    # graph_path = train_kwargs['graph_path']
    log_path = train_kwargs['log_path']
    budget = train_kwargs['budget']
    dataset = train_kwargs['dataset']
    data_config = train_kwargs.get('data_config')
    active_data_config = data_config or community_data_config
    data_experiment = (
        Path(active_data_config).stem
        if active_data_config else 'legacy_single_source'
    )
    seed = int(train_kwargs['seed'])

    train_kwargs['return_mode'] = 'full'
    del train_kwargs['trial']
    torch_n_threads = train_kwargs.pop('torch_n_threads', torch_n_threads)

    # Offline training
    log_dir_path = os.path.join(
        log_root, log_path, budget, dataset, data_experiment, f"seed_{seed}"
    )
    data_saving_dir = os.path.join(
        train_kwargs['data_saving_dir'], budget, dataset, data_experiment,
        f"seed_{seed}"
    )
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(data_saving_dir, exist_ok=True)
    train_kwargs['data_saving_dir'] = data_saving_dir
    run_metadata = {
        "dataset": dataset,
        "budget_percent": float(budget),
        "data_experiment": data_experiment,
        "data_config": data_config,
        "community_data_config": community_data_config,
        "graph_path": train_kwargs['graph_path'],
        "training_seed": seed,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_id": train_kwargs.get('gpu_id'),
        "n_epochs": train_kwargs['n_epochs'],
        "n_grad_steps": train_kwargs['n_grad_steps'],
        "minibatch_size": train_kwargs['minibatch_size'],
        "policy_lr": float(train_kwargs['policy_lr']),
        "value_lr": float(train_kwargs['value_lr']),
        "alpha_lr": float(train_kwargs['alpha_lr']),
        "algorithm_mode": train_kwargs['algorithm_mode'],
        "beta": float(train_kwargs['beta']),
        "cql_alpha": float(train_kwargs['cql_alpha']),
        "gnn_backbone": train_kwargs['gnn_backbone'],
        "gnn_depth": int(train_kwargs['gnn_depth']),
        "gat_heads": int(train_kwargs['gat_heads']),
        "paper_architecture": bool(train_kwargs.get('paper_architecture', False)),
        "use_mlp": bool(train_kwargs['use_MLP']),
        "selection_mode": train_kwargs['inference_selection_mode'],
        "num_partitions": train_kwargs['inference_num_partitions'],
        "partition_seed": train_kwargs['inference_partition_seed'],
        "status": "running",
    }
    metadata_path = os.path.join(data_saving_dir, "training_run.json")
    with open(metadata_path, "w") as file:
        json.dump(run_metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    if torch.cuda.is_available() and int(train_kwargs.get('gpu_id', -1)) >= 0:
        gpu_id = int(train_kwargs['gpu_id'])
        # torch 1.7 can segfault in reset_peak_memory_stats before the CUDA
        # context exists.  Explicit initialization makes resource accounting
        # safe and also fails early with a normal CUDA error if the device is
        # unavailable.
        torch.cuda.set_device(gpu_id)
        torch.empty(1, device=torch.device('cuda:' + str(gpu_id)))
        torch.cuda.reset_peak_memory_stats(gpu_id)
    training_started = time.time()
    try:
        full_score = train_agent(train_func,
                                 log_dir=log_dir_path,
                                 train_kwargs=train_kwargs,
                                 snapshot_frequency=snapshot_frequency,
                                 x_axis='Epoch')
    except Exception as error:
        run_metadata["status"] = "failed"
        run_metadata["error"] = repr(error)
        run_metadata["elapsed_seconds"] = time.time() - training_started
        run_metadata["peak_rss_mb"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        )
        with open(metadata_path, "w") as file:
            json.dump(run_metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        raise
    checkpoints = sorted(
        Path(data_saving_dir).glob("epoch_*.pkl"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    run_metadata["status"] = "completed"
    run_metadata["elapsed_seconds"] = time.time() - training_started
    run_metadata["peak_rss_mb"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    )
    if torch.cuda.is_available() and int(train_kwargs.get('gpu_id', -1)) >= 0:
        gpu_id = int(train_kwargs['gpu_id'])
        run_metadata["peak_cuda_allocated_mb"] = (
            torch.cuda.max_memory_allocated(gpu_id) / (1024.0 ** 2)
        )
        run_metadata["peak_cuda_reserved_mb"] = (
            torch.cuda.max_memory_reserved(gpu_id) / (1024.0 ** 2)
        )
    run_metadata["checkpoint_count"] = len(checkpoints)
    run_metadata["final_checkpoint"] = str(checkpoints[-1].resolve()) if checkpoints else None
    with open(metadata_path, "w") as file:
        json.dump(run_metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    
    print("full_score==", full_score)

    # window = 10  # 每隔多少个epoch输出一次performance score 的中值
    # # score = np.median(full_score[-min(len(full_score), window):])
    #
    # score = np.median(full_score[-min(len(full_score), window):])
    #
    # print('Median of performance of last {} epochs'.format(window), score)
    # return {'score': score,  # last 50 epochs
    #         'mean': np.mean(full_score)}


if __name__ == '__main__':
    import argparse
    from atac.garage_tools.utils import str2bool
    # 设置更高的整数字符串转换限制
    sys.set_int_max_str_digits(10000)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--discount', type=float, default=0.9)
    
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--force_cpu_data_collection', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    # 热启动建议直接设置为0，不需要有热启动步骤
    parser.add_argument('--n_warmstart_steps', type=int, default=0)
    parser.add_argument('--fixed_alpha', type=float, default=None)
    # 批评网络的参数，可以作为新的消融验证
    parser.add_argument('--beta', type=float, default=2)
    parser.add_argument('--norm_constraint', type=float, default=1000)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--value_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_lr', type=float, default=5e-5)
    parser.add_argument('--target_update_tau', type=float, default=5e-3)
    parser.add_argument('--minibatch_size', type=int, default=8)
    parser.add_argument('--use_deterministic_evaluation', type=str2bool, default=True)
    # 使用双Q，会多一个qf2，和qf1的参数，以及更新方式一样
    parser.add_argument('--use_two_qfs', type=str2bool, default=True)
    # 消融参数，应该设置为0，1
    parser.add_argument('--q_eval_mode', type=str, default='0.5_0.5')
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--n_grad_steps', type=int, default=500)
    parser.add_argument('--use_policy_lr_scheduler', type=str2bool, default=False)
    # parser.add_argument('--use_critic_lr_scheduler', type=str2bool, default=False)
    parser.add_argument('--use_std_loss', type=str2bool, default=False)
    # parser.add_argument('--use_resource_reward', type=str2bool, default=True)
    parser.add_argument('--use_weighted_adj', type=str2bool, default=False)
    parser.add_argument('--use_MLP', type=str2bool, default=False)

    # 可修改的参数区间
    parser.add_argument('--n_epochs', type=int, default=100)

    # 存代码的参数
    parser.add_argument('--log_path', type=str, default='log11-11')
    parser.add_argument('--gpu_id', type=int, default=0)  # use cpu by default
    parser.add_argument('--data_saving_dir', type=str, default=str(DEFAULT_RUN_ROOT / 'artifacts'))
    parser.add_argument('--dataset', type=str, default='epinions')
    parser.add_argument('--budget', type=str, default='5')
    parser.add_argument('--data_path1', type=str, default=None)
    parser.add_argument('--data_config', type=str, default=None,
                        help='JSON manifest for reproducible offline-data mixing')
    parser.add_argument('--community_data_config', type=str, default=None,
                        help='heterogeneous Leiden-community offline manifest')
    parser.add_argument('--graph_path', type=str, default=None)
    parser.add_argument('--inference_selection_mode', type=str, default='partitioned',
                        choices=('partitioned', 'global_topk'))
    parser.add_argument('--inference_num_partitions', type=int, default=None)
    parser.add_argument('--inference_partition_seed', type=int, default=42)
    parser.add_argument('--algorithm_mode', choices=('atac', 'vanilla_ac', 'cql'),
                        default='atac')
    parser.add_argument('--cql_alpha', type=float, default=1.0)
    parser.add_argument('--gnn_backbone', choices=('gcn', 'gat', 'sage'),
                        default='gcn')
    parser.add_argument('--gnn_depth', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=1)
    parser.add_argument('--paper_architecture', type=str2bool, default=False)
    parser.add_argument('--torch_n_threads', type=int, default=2)
    train_kwargs = vars(parser.parse_args())
    run(**train_kwargs)
