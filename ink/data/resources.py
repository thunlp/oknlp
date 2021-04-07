import pkg_resources
import os, yaml
from typing import List

RESOURCES_MAP = {}

def generate_filelist(obj, prefix="") -> List[str]:
    if not isinstance(obj, list):
        raise ValueError("Parse config file failed", obj)
    ret = []
    for it in obj:
        if isinstance(it, str):
            ret.append( os.path.join(prefix, it) )
        elif isinstance(it, dict) and ("dir" in it):
            ret.append( os.path.join(prefix, it["dir"]) )
            if "files" in it:
                ret.extend( generate_filelist(it["files"], os.path.join(prefix, it["dir"])) )
        else:
            raise ValueError("Parse config file failed", it)
    return ret

def init(res_dir_name):
    resources = pkg_resources.resource_listdir(__name__, res_dir_name)
    for resource_def_file_name in resources:
        v = yaml.load(pkg_resources.resource_stream(__name__, os.path.join(res_dir_name, resource_def_file_name)), Loader=yaml.FullLoader)
        if v["name"] not in RESOURCES_MAP:
            RESOURCES_MAP[v["name"]] = []
        RESOURCES_MAP[v["name"]].append({
            "name": v["name"],
            "config": resource_def_file_name,
            "version": v["version"],
            "path": v["path"],
            "files": generate_filelist(v["files"])
        })

def get_resouce_info(resource_name, version=None):
    if resource_name not in RESOURCES_MAP:
        raise ValueError("Unknown resource name `%s`" % resource_name)
    if version is not None:
        for obj in RESOURCES_MAP[resource_name]:
            if obj["version"] == version:
                return obj
        raise ValueError("Version `%s` for resource `%s` is not found" % (version, resource_name))
    else:
        all_int = True
        mx_int = None
        ret = None
        for obj in RESOURCES_MAP[resource_name]:
            if  not isinstance(obj["version"], int):
                all_int = False
            else:
                if mx_int is None or mx_int < obj["version"]:
                    mx_int = obj["version"]
                    ret = obj
        if all_int:
            if ret is None:
                raise ValueError()
            else:
                return ret
        elif len(RESOURCES_MAP[resource_name]) > 0:
            return RESOURCES_MAP[resource_name][-1]
        else:
            raise ValueError()

init("resources")