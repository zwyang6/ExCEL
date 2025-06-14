from model.model_excel import ExCEL_model

def build_network(args):

    model = ExCEL_model(
                        clip_model=args.model, embedding_dim=args.embedding_dim, in_channels=args.in_channels, \
                        dataset_name=args.dataset_name, \
                        num_classes=args.num_classes, num_atrr_clusters=args.num_attri, json_file=args.attr_json,\
                        img_size=args.crop_size, mode=args.train_set, device='cuda')
    param_groups = model.get_param_groups()

    return model, param_groups