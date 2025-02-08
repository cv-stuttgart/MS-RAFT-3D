import json


mandatory = {}


def set_defaults(config):
    default_values = {}
    default_values["name"] = mandatory
    default_values["checkpoint_save_path"] = None
    default_values["checkpoint_load_path"] = None
    default_values["feature_encoder"] = "ccmr"
    default_values["context_encoder"] = mandatory
    default_values["context_encoder_dim"] = None
    default_values["corr"] = "pre_calc"
    default_values["corr_levels"] = 4
    default_values["corr_radius"] = 3
    default_values["hidden_dim"] = 128
    default_values["context_dim"] = 384
    default_values["update_operator"] = "bilaplacian"
    default_values["se3_lm"] = .0001
    default_values["se3_ep"] = 10.0
    default_values["se3_neighborhood"] = 32
    default_values["iterations"] = [4, 6, 8]
    default_values["adamw_eps"] = 1e-8
    default_values["grad_acc"] = 1
    default_values["gpus"] = [0]
    default_values["train_save_vram"] = False
    default_values["train"] = {}
    default_values["train"]["num_steps"] = mandatory
    default_values["train"]["lr"] = mandatory
    default_values["train"]["dataset"] = mandatory
    default_values["train"]["batch_size"] = mandatory
    default_values["train"]["image_size"] = mandatory
    default_values["train"]["wdecay"] = mandatory
    default_values["train"]["gamma"] = mandatory
    default_values["train"]["loss_fn"] = mandatory
    default_values["initial_phase"] = 0
    default_values["initial_step"] = 0

    for key, value in config.items():
        if key not in default_values:
            raise ValueError(f" Argument {key} was set but is unknown")
        default_values[key] = value

    for key, value in default_values.items():
        if value is mandatory:
            raise ValueError(f" Manditory argument {key} was set not set")

    return default_values


def load_config(args):
    file = open(args.config)
    config = json.load(file)

    config["checkpoint_load_path"] = getattr(args, "ckpt", None)
    config["checkpoint_save_path"] = getattr(args, "save", None)
    config["initial_phase"] = getattr(args, "initial_phase", 0)
    config["initial_step"] = getattr(args, "initial_step", 0)

    return set_defaults(config)


# def cpy_eval_args_to_config(args):
#     config = {}
#     config["model"] = args.model
#     config["disp_kitti_path"] = args.disp_kitti_path
#     config["eval_fth_path"] = args.eval_fth_path
#     config["initial_phase"] = args.initial_phase
#     config["initial_step"] = args.initial_step

#     return config
