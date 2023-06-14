"""Patch pyyaml."""

import yaml


# Due to issue
# https://github.com/yaml/pyyaml/issues/165
# use the workaround to raise on duplicate YAML keys from
# https://gist.github.com/pypt/94d747fe5180851196eb?permalink_comment_id=4015118#gistcomment-4015118
# on 2023-03-27.
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def safe_load(string):
    return yaml.load(string, Loader=UniqueKeyLoader)
