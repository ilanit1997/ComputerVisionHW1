from roboflow import Roboflow
# import train.py



def load_data():
    rf = Roboflow(api_key="mA7koMgDVFBLYkhBiJtz")
    project = rf.workspace("ilanitdavidcv1").project("cv_hw1_flipped_dataset")
    dataset = project.version(1).download("yolov5")
    return dataset


def define_model_config(dataset):
    """
    We will write a yaml script that defines the parameters for our model
    like the number of classes, anchors, and each layer.
    :return:
    """
    # define number of classes based on YAML
    import yaml
    with open(dataset.location + "/data.yaml", 'r') as stream:
        num_classes = str(yaml.safe_load(stream)['nc'])




if __name__ == "__main__":
    ds = load_data()
    define_model_config(ds)