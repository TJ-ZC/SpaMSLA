def params(args):
    if args.data_type == 'HLN_A1':
        args.learning_rate, args.weight_decay = 0.01, 5e-3
        args.epochs, args.weight1, args.weight2, args.weight3 = 700, 4, 4, 4
        args.n_clusters = 10
        args.file_fold = './Data/HLN/'
        args.hidden_feat, args.dropout = 128, 0.3

    elif args.data_type == 'CITE':
        args.learning_rate, args.weight_decay = 0.01, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 700, 2, 2, 6
        args.n_clusters = 8
        args.file_fold = './Data/COSMOS/'
        args.hidden_feat, args.dropout = 64, 0.1

    elif args.data_type == 'COSMOS-ATAC-RNA-seq':
        args.learning_rate, args.weight_decay = 0.001, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 1100, 2, 19, 8
        args.n_clusters = 13
        args.file_fold = './Data/COSMOS/'
        args.hidden_feat, args.dropout = 64, 0.4
        args.tool = "leiden"

    elif args.data_type == 'H3K27ac':
        args.learning_rate, args.weight_decay = 0.001, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 1400, 1, 10, 20
        args.n_clusters = 14
        args.file_fold = './Data/Mouse_Brain_H3K27ac/'
        args.hidden_feat, args.dropout = 64, 0.3
        args.tool = "leiden"

    elif args.data_type == 'H3K27me3':
        args.learning_rate, args.weight_decay = 0.001, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 1400, 1, 10, 20
        args.n_clusters = 14
        args.file_fold = './Data/Mouse_Brain_H3K27me3/'
        args.hidden_feat, args.dropout = 64, 0.3
        args.tool = "leiden"

    elif args.data_type == 'H3K4me3':
        args.learning_rate, args.weight_decay = 0.001, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 1400, 1, 10, 20
        args.n_clusters = 14
        args.file_fold = './Data/Mouse_Brain_H3K4me3/'
        args.hidden_feat, args.dropout = 64, 0.3
        args.tool = "leiden"

    elif 'MISAR' in args.data_type:
        args.file_fold = './Data/MISAR_seq/'
        args.hidden_feat, args.dropout = 128, 0.2
        args.learning_rate, args.weight_decay = 0.01, 5e-2
        if args.data_type == 'MISAR7':
            args.epochs, args.weight1, args.weight2, args.weight3 = 200, 1, 17, 11
            args.n_clusters = 7
        elif args.data_type == 'MISAR12':
            args.epochs, args.weight1, args.weight2, args.weight3 = 100, 1, 14, 5
            args.n_clusters = 12

    elif 'SPOTS' in args.data_type:
        args.hidden_feat, args.dropout = 64, 0.1
        args.learning_rate, args.weight_decay = 0.01, 5e-3
        args.n_clusters = 5
        if args.data_type == 'SPOTS1':
            args.file_fold = './Data/Mouse_Spleen1/'
            args.epochs, args.weight1, args.weight2, args.weight3 = 600, 1, 2, 4
        elif args.data_type == 'SPOTS2':
            args.file_fold = './Data/Mouse_Spleen2/'
            args.epochs, args.weight1, args.weight2, args.weight3 = 600, 1, 3, 2

    elif args.data_type == 'STARmap':
        args.learning_rate, args.weight_decay = 0.001, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 100, 1, 3, 7
        args.n_clusters = 6
        args.file_fold = './Data/simu_STARmap/'
        args.hidden_feat, args.dropout = 64, 0.1
        args.tool = 'leiden'

    elif args.data_type == 'Stereo_simu':
        args.learning_rate, args.weight_decay = 0.001, 5e-2
        args.epochs, args.weight1, args.weight2, args.weight3 = 700, 1, 12, 14
        args.n_clusters = 6
        args.file_fold = './Data/simu_Stereo_seq/'
        args.hidden_feat, args.dropout = 128, 0.2

    return args
