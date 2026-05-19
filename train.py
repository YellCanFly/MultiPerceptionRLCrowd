from config import parse_args
from train_strategy import (
    # entity
    train_single_individual_entity_net,
    train_single_individual_entity_weight_net,
    # implicit
    train_single_individual_pure_implicit_net,
    train_single_individual_pure_implicit_weight_net,
    # combine feature
    train_single_individual_feature_combine_net,
    train_single_individual_feature_combine_weight_net,

    # ablation study
    train_single_individual_holistic_net,
    train_single_individual_holistic_net_no_weight_mlp,
)

if __name__ == '__main__':
    args = parse_args()
    #---------只训练Entity模块--------#
    if args.train_strategy == 'single_individual_entity_net':
        train_single_individual_entity_net.main()
    elif args.train_strategy == 'single_individual_entity_weight_net':
        train_single_individual_entity_weight_net.main()

    # ---------只训练Implicit模块--------#
    elif args.train_strategy == 'single_individual_pure_implicit_net':
        train_single_individual_pure_implicit_net.main()
    elif args.train_strategy == 'single_individual_pure_implicit_weight_net': # 只训练implicit, 且有weight mlp
        train_single_individual_pure_implicit_weight_net.main()

    # ---------将预训练的Entity和Implicit模块结合训练--------#
    elif args.train_strategy == 'single_individual_feature_combine_net':
        train_single_individual_feature_combine_net.main()
    elif args.train_strategy == 'single_individual_feature_combine_weight_net':
        train_single_individual_feature_combine_weight_net.main()

    # ---------Ablation Study相关实验--------#
    elif args.train_strategy == 'single_individual_holistic_net':
        train_single_individual_holistic_net.main()
    elif args.train_strategy == 'single_individual_holistic_net_no_weight_mlp':
        train_single_individual_holistic_net_no_weight_mlp.main()

    #---------报错说明--------#
    else:
        print('train strategy type error!')
