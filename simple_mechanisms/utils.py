import os, traceback, importlib
from typing import Any, Callable, Tuple, Iterable, Dict, Union, Optional, Sequence

class ConfigException(Exception):
    @staticmethod
    def filter_traceback(tb: Iterable[traceback.StackSummary], config_file_path: str) -> Iterable[traceback.StackSummary]:
        assert config_file_path is not None
        config_file_name = os.path.basename(config_file_path)
        encountered_config = False
        for frame in tb:
            if frame.filename.endswith(config_file_name):
                encountered_config = True
            if encountered_config:
                yield frame

    def __init__(self, title: str, msg: str=None, base_exception: Optional[Exception]=None) -> None:
        self.msg = msg if msg is not None else "Uncategorised error"
        self.base = base_exception
        self.title = title

        super().__init__(self.format_message())

    def format_message(self, config_file_path: Optional[str]=None) -> str:
        full_message = f"\n#### {self.title.upper()} ####\n##   {self.msg}"
        if self.base is not None:
            full_message += f"\nTechnical error description below:\n"
            tb = traceback.extract_tb(self.base.__traceback__)
            if config_file_path is not None:
                tb = self.filter_traceback(tb, config_file_path)
            full_message += "\n".join(traceback.format_list(tb))
            full_message += "\n".join(traceback.format_exception_only(type(self.base), self.base))

        return full_message

class ConfigParsingException(ConfigException):
    def __init__(self, msg: str, base_exception: Optional[Exception]=None) -> None:
        self.msg = msg
        self.base = base_exception

        title = "FAILED TO PARSE THE CONFIG"

        super().__init__(title, msg, base_exception)

class ConfigParsingUnknownException(ConfigParsingException):
    def __init__(self, function_name: str, base_exception: Exception) -> None:
        self.__init__(f"Uncategorised error while trying to access function '{function_name}' from config module.", base_exception)

def load_configuration(config_module_path):
    if not os.path.exists(config_module_path):
        raise FileNotFoundError(config_module_path)

    try:
        spec = importlib.util.spec_from_file_location("config_module", config_module_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
    except Exception as e: # handling errors in py-file parsing
        raise ConfigParsingException("Unable to read the specified file as a Python module.", e) from e
    
    try: config = config_module.config
    except AttributeError:
        raise ConfigParsingException("Config module does not specify a 'config' function")

    return config
