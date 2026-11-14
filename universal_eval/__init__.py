# -*- coding: utf-8 -*-
"""
universal_eval — 五层解耦通用模型推理评测框架

架构分层:
  data/         — 数据层：下载、加载、预处理评测数据
  model/        — 模型层：下载、部署、调度模型推理
  algorithm/    — 算法层：推理算法撰写与调度
  metric/       — 指标层：按任务类型配置不同评测指标
  orchestrator/ — 中枢层：端到端任务管线配置与执行
"""

__version__ = "2.0.0"