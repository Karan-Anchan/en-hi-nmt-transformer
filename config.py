def get_config():
    return {
        "batch_size" : 8,
        "num_epochs" : 50,
        "lr" : 10**-4,
        "seq_len": 256,
        "d_model": 512,
        "lang_src" : "en",
        "lang_tgt" : "hi",
        "model_folder" : "weights",
        "model_filename" : "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel"
    }
    
