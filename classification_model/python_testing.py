import yaml

with open("config.yaml", "r") as f:
    try:
        config_file = yaml.safe_load(f)
        print(config_file["data_link"])
    except yaml.YAMLError as exc:
        print(exc)