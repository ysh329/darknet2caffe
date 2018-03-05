def to_old_api(inferx_model_c):

    new_api_dict = {"inferx_finalize": "Finalize",\
                    "inferx_convolution": "Convolution",\
                    "inferx_relu": "ReLU",\
                    "inferx_batchnorm": "BatchNorm",\
                    "inferx_scale": "Scale",\
                    "inferx_pooling_yolo": "Pooling_yolo",\
                    "inferx_pooling": "Pooling",\
                    "inferx_reshape": "Reshape",\
                    "inferx_concat": "Concat",\
                    "inferx_input": "Input",\
                    "inferx_parse_str": "parseStr",\
                    "inferx_set_init_var": "setInitVar",\
                    "inferx_var_add_init": "varAddInit",}

    with open(inferx_model_c, "r") as file_handle:
        model_c_lines_str = file_handle.read()

    # replace
    for new_api in new_api_dict:
        model_c_lines_str = model_c_lines_str.replace(new_api, new_api_dict[new_api])

    with open(inferx_model_c, "w") as file_handle:
        file_handle.write(model_c_lines_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python %s INFERX_MODEL_C\n" % sys.argv[0])
        exit(-1)

    inferx_model_c = sys.argv[1]
    to_old_api(inferx_model_c)
    print("Successful conversion to old inferx model api")

