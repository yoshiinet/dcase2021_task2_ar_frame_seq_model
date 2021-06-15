# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
from common import com, TargetDomain
from data.task2_basedata import BaseData

class FlowItem(object):
    machine_type_list = ['ToyCar','ToyTrain','fan', 'gearbox', 'pump','slider','valve']
    target_dir_list = ['source_test', 'target_test']

    def __init__(self, machine_type, section_index, target):
        self.domain = TargetDomain(machine_type, section_index, target)

    def model_file_path(self, purpose, target, eval_key):
        return com.model_file_path(self.domain, purpose, target, eval_key)

    def score_distr_param_path(self, target=None):
        return com.score_distr_param_path(self.domain, target=target)

    def score_distr_csv_path(self, target=None):
        return com.score_distr_csv_path(self.domain, target=target)

    def score_distr_fig_path(self, target=None):
        return com.score_distr_fig_path(self.domain, target=target)

    def history_img_path(self):
        return com.history_img_path(self.domain)

    def history_json_path(self):
        return com.history_json_path(self.domain)

    def base_data(self, target_dir, augment, fit_key, eval_key):
        """
        target_dir: one of ['train','source_test','target_test']
        """
        assert target_dir in ['train','source_test','target_test']
        return BaseData(machine_type=self.domain.machine_type,
                        section_name='section_'+self.domain.section_index,
                        domain=self.domain.target,
                        target_dir=target_dir, augment=augment,
                        fit_key=fit_key, eval_key=eval_key)

    def test_items(self):
        """
        評価する対象を生成する

        return: a tuple, (machine_type, section_name, target_dir)
            machine_type : eg. ToyCar, etc.
            target_dir     : eg. source_test, etc.
            section_name : eg. source_00, etc.
            
        """
        for machine_type in self.machine_types(): # このitemで評価するmachine_types
            machine_dir = com.machine_dir(machine_type) # eg. .../dev_data/ToyCar
            for target_dir in self.target_dirs(): # one of ['source_test', 'target_test']
                for section_name in self.section_names(machine_dir, target_dir):
                    yield machine_type, section_name, target_dir

    def machine_types(self):
        """
        yield machine_type, for test, depend on self.domain

        machine_type : one of ['ToyCar','ToyTrain','fan', 'gearbox', 'pump','slider','valve']
        """
        machine_spec = self.domain.machine_type # 'ToCar' or 'ToTrain' or more

        usable_machine_types = com.param['limit']['usable_machine_types']
        for machine_type in self.machine_type_list:
            if usable_machine_types and machine_type not in usable_machine_types:
                continue # skip
            if machine_spec == '*' or machine_type == machine_spec:
                yield machine_type

    def section_names(self, machine_dir, target_dir):
        """
        yield section_name, for test, depend on self.domain

        section_name : one of ['section_00', 'section_01', 'section_02', etc.]
        """
        section_spec = self.domain.section_index # '00' or '01' or '02' or more

        section_names = com.get_section_names(machine_dir, target_dir=target_dir)
        for section_name in section_names:
            if section_spec == '*' or section_name.endswith(section_spec):
                yield section_name

    def target_dirs(self):
        """
        yield target_dir, for test, depend on self.domain

        target_dir : one of ['source_test', 'target_test']
        """
        target_spec = self.domain.target # 'source' or 'target' or '*'

        for target_dir in self.target_dir_list:
            if target_spec == '*' or target_dir.startswith(target_spec):
                yield target_dir 
