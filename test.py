import importlib
from config.template import TemplateConfig

def load_config(module_path: str, class_name: str, **kwargs):
    # Import the module dynamically
    module = importlib.import_module(module_path)
    # Get class by reflection
    config_class = getattr(module, class_name)
    return config_class()

if __name__ == "__main__":
    config:TemplateConfig = load_config('config.cifar10', 'Config')
    print(config)
    print(config.batch_size)
    pass