# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
from flow.for_any import FlowDependNone
from flow.for_machine import FlowDependMachine
from flow.for_machine_target import FlowDependMachineTarget
from flow.for_machine_section import FlowDependMachineSection
from flow.for_machine_section_target import FlowDependMachineSectionTarget

def get_flow_instance(param):
    """
    return a flow instance
    """
    f_machine = param['model_for']['f_machine']
    f_section = param['model_for']['f_section']
    f_target = param['model_for']['f_target']

    if f_machine:
        if f_section:
            if f_target:
                return FlowDependMachineSectionTarget(f_target)
            else:
                return FlowDependMachineSection(f_target)
        else:
            if f_target:
                return FlowDependMachineTarget(f_target)
            else:
                return FlowDependMachine(f_target)
    else:
        return FlowDependNone(f_target)
