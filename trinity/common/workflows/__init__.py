# -*- coding: utf-8 -*-
"""Workflow module"""
from trinity.common.workflows.workflow import Task, Workflow
from trinity.utils.registry import Registry

from trinity.common.workflows.envs import downward_pool

downward_pool.install()  # ALFWorld: reuse fast_downward library copies (tmpfs leak)

WORKFLOWS: Registry = Registry(
    "workflows",
    default_mapping={
        # simple/math
        "simple_workflow": "trinity.common.workflows.workflow.SimpleWorkflow",
        "async_simple_workflow": "trinity.common.workflows.workflow.AsyncSimpleWorkflow",
        "math_workflow": "trinity.common.workflows.workflow.MathWorkflow",
        "async_math_workflow": "trinity.common.workflows.workflow.AsyncMathWorkflow",
        # LookAheadOPD (envs/lookahead_opd/; the gate itself runs in the tunix trainer)
        "LookAheadOPD_alfworld_workflow": "trinity.common.workflows.envs.lookahead_opd.alfworld.LookAheadOPDAlfworldWorkflow",
        "LookAheadOPD_scienceworld_workflow": "trinity.common.workflows.envs.lookahead_opd.scienceworld.LookAheadOPDScienceworldWorkflow",
        "LookAheadOPD_webshop_workflow": "trinity.common.workflows.envs.lookahead_opd.webshop.LookAheadOPDWebshopWorkflow",
    },
)

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
]
