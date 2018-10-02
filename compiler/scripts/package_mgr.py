import os
import logging
import tarfile
import json
import argparse

'''
This module looks for def.json in a given path and identifies the (.bin, .json
and .npy) files to be packed as part of kelf.tar.gz.
Example def.json:
    {
    "act": "act.json",
    "act_instr": "TrivNet-act.bin",
    "dma_queue": {
        "IN_QUE": {
            "type": "in"
        },
        "q_act_out": {
            "owner": "act",
            "type": "out"
        },
        "q_pe_in_w": {
            "owner": "pe",
            "type": "data"
        },
        "q_pool_out": {
            "owner": "pool",
            "type": "out"
        }
    },
    "host": "host.json",
    "name": "definition",
    "pe": "pe.json",
    "pe_instr": "TrivNet-pe.bin",
    "pool": "pool.json",
    "pool_instr": "TrivNet-pool.bin",
    "var": {
        "IN": {
            "#transfer-type": "input",
            "size": 301056,
            "type": "input"
        },
        "SB": {
            "type": "state-buffer"
        },
        "W0": {
            "#transfer-type": "weight",
            "file_name": "trivnet_conv1__kernel:0_CRSM.npy",
            "type": "file"
        },
        "W1": {
            "#transfer-type": "weight",
            "file_name": "trivnet_bn_conv1__batchnorm_1__sub___104__cf__104:0_NCHW.npy",
            "type": "file"
        },
    "version": "0.1-bcfd4f2"
}
'''

engine_list = ['pe', 'pool', 'act']
json_file_list = ['host', 'pe', 'pool', 'act']
class KPackageManager(object):
    
    def __init__(self, manifest):
        self.logger = logging.getLogger('KelfPkgMgr')
        self.manifest = manifest
        self.kout_name        = 'kelf.tar.gz'
        self.manifest_file_name    = 'def.json'
        self.json_suffix      = '.json'
        self.bin_suffix       = '.bin'
        self.numpy_suffix     = '.npy'
        self.instr_key_suffix = '_instr'
        self.var_key          = 'var'
        self.weight_key       = 'file_name'
        self.error_codes      = {'MISSING_ENTRY': 1, 'INVALID_FORMAT': 2,
                                 'MISSING_FILE': 3, 'INVALID_ARG': 4, 'MISSING_FILE': 5}
        self.files_to_pack    = []


    def __check_and_get_numpy_files_list(self, numpy_files):
        if self.var_key not in self.manifest.keys():
            self.logger.error('{} entry not found in manifest file..'.format(self.var_key))
            return self.error_codes['MISSING_ENTRY']

        for entry_k, entry_v in self.manifest[self.var_key].items():
            if type(entry_v) == dict and self.weight_key in entry_v.keys():
                if entry_v[self.weight_key].endswith(self.numpy_suffix):
                    numpy_files.append(entry_v[self.weight_key])
                else:
                    self.logger.error('Invalid file format for {}:{}'.format(ententry_k, entry_v[self.manifest[weight_key]]))
                    return self.error_codes['INVALID_FORMAT']
        return 0
             

    def get_files(self):
        ''' scans throgh the manifest dict and checks for all the required
        files.'''
        #check if the key exists for each engine and its pointing to .json file
        for eng in json_file_list:
            if eng in self.manifest.keys():
                if not self.manifest[eng].endswith(self.json_suffix):
                    self.logger.error('Invalid file format {}:{}'.format(eng, self.manifest[eng]))
                    return self.error_codes['INVALID_FORMAT']
            else:
                self.logger('Missing entry for {}', eng)
                return self.error_codes['MISSING_ENTRY']
            self.files_to_pack.append(self.manifest[eng])
        #check if the instr key exists and its pointing to .bin file
        for eng in engine_list:
            instr_key = eng + self.instr_key_suffix
            if instr_key in self.manifest.keys():
                if not self.manifest[instr_key].endswith(self.bin_suffix):
                    self.logger('Invalid file format for {}:{}'.format(instr_key, self.manifest[instr_key]))
                    return self.error_codes['INVALID_FORMAT']
            else:
                self.logger.error('MIssing entry for {}', instr_key)
                return self.error_codes['MISSING_ENTRY']
            self.files_to_pack.append(self.manifest[instr_key])

        numpy_files_list = []
        ret = self.__check_and_get_numpy_files_list(numpy_files_list)
        if ret:
            self.logger.error('failed to get list of numpy files...')
            return ret
        if len(numpy_files_list) == 0:
            self.logger.error('No numpy files found...')
            return self.error_codes['MISSING_ENTRY']
        self.files_to_pack.extend(numpy_files_list)
        self.files_to_pack.append(self.manifest_file_name)

        return 0


    def __check_if_files_exist(self):
       is_file_exist = [os.path.isfile(f) for f in self.files_to_pack]
       if not all(is_file_exist):
           missing_files = [f for f, is_present in zip(self.files_to_pack, is_file_exist) if is_present == False]
           self.logger.error('missing files: {}'.format(missing_files))
           return self.error_codes['MISSING_FILE']
       return 0


    def gen_pkg(self, path=''):
        working_dir = None
        if path == '':
            working_dir = os.getcwd()
        else:
            working_dir = path
            if not os.path.isdir(working_dir):
                self.logger.error('{} does not exist...')
                return self.error_codes['INVALID_ARG']
        self.logger.debug('Looking for model files under {}', working_dir)
        os.chdir(working_dir)
        print('working dir: {}'.format(working_dir)) 
        ret = self.get_files()
        if ret:
            self.logger.error('failed to get files list..')
            return ret
        ret = self.__check_if_files_exist()
        if ret:
            self.logger.error('Missing files...')
            return ret
        with tarfile.open(self.kout_name, 'w:gz') as tar:
            for f in self.files_to_pack:
                tar.add(f)
        print("{} file generated successfully...".format(self.kout_name))
        return 0


def create_kelf(model_dir, subgraphs, output_file=None):
    """ Creates kelf package(tar.gz) for give subgraphs.

        Walks all subgraph directories and gathers file list.
        Creates model.json to describe the NN.
        Create tar.gz with all required files.
    """
    assert subgraphs
    assert os.path.isdir(model_dir)

    if output_file is None:
        output_file = os.path.join(model_dir, 'kelf.tar.gz')

    model_info = {}
    for sg in subgraphs:
        sg_name = sg['SubGraphDir']
        sg_info = {
            'executor': sg['executor'],
            'inputs': [sg_input['name'] for sg_input in sg['Inputs']],
            'outputs': [sg_output['name'] for sg_output in sg['Outputs']]
        }
        model_info[sg_name] = sg_info

    model_json = os.path.join('model.json')
    with open(model_json, 'w') as f:
        f.write(json.dumps(model_info, indent=4))

    files_to_pack = []
    for sg in model_info:
        sg_dir = os.path.join(model_dir, sg)
        assert os.path.isdir(sg_dir)
        manifest_file = os.path.join(sg_dir, 'def.json')
        # host sg's wont have def.json - silently ignore them till runtime supports it.
        if not os.path.isfile(manifest_file):
            continue
        with open(manifest_file, 'r') as fd:
            manifest = json.loads(fd.read())
        sg_pkg = KPackageManager(manifest)
        ret = sg_pkg.get_files()
        # Ignore errors since "host" executor wont have all the files.
        if ret != 0:
            continue
        for f in sg_pkg.files_to_pack:
            files_to_pack.append(os.path.join(sg, f))

    with tarfile.open(output_file, 'w:gz') as tar:
        for f in files_to_pack:
            tar.add(os.path.join(model_dir, f), arcname=f)
        tar.add(model_json)
    print('Created {}'.format(output_file))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', help='Path to def.json of a single subgraph(packages only single subgraph).')
    parser.add_argument('--model', help='Location of a model directory(package all subgraphs in the model).')

    args = parser.parse_args()

    if args.manifest:
        manifest = {}
        with open(args.manifest, 'r') as fd:
            manifest = json.loads(fd.read())
        pkg_mgr = KPackageManager(manifest)
        pkg_mgr.gen_pkg()
    else:
        nn_json = os.path.join(args.model, 'nn_graph.json')
        with open(nn_json, 'r') as fd:
            json_file = json.loads(fd.read())
            subgraphs = json_file['SubGraphs']
        create_kelf(args.model, subgraphs)

if __name__ == '__main__':
    main()
