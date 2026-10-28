#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成「vLLM V1 一次调度（一个 EngineCore.step）到底做了什么」单页图（.drawio，未压缩 XML）。

口径锚定本仓库 share/2026Q4/vllm main 00b7847c80（≈ v0.30.1rc0-60）：
  vllm/v1/engine/core.py:633-662          EngineCore.step()：schedule → execute_model → update_from_output
  vllm/v1/core/sched/scheduler.py:2485    Scheduler.add_request()：进 waiting 队列 + 记状态
  vllm/v1/core/sched/scheduler.py:556      Scheduler.schedule()：token_budget 初始化
  vllm/v1/core/sched/scheduler.py:623-866  第一段：RUNNING 请求 + allocate_slots + 抢占
  vllm/v1/core/sched/scheduler.py:867-1389 第二段：WAITING 请求（有抢占则整段跳过）
  vllm/v1/core/sched/scheduler.py:1461-1483 SchedulerOutput 组装
  vllm/v1/core/sched/scheduler.py:1918     update_from_output()：追加 token / 判停止 / 释放 KV

版式刻意保留 docs/流程图.jpg 的「竖排流程 + 箭头 + 浅绿高亮框」骨架（便于和原图对照），
但把三处不准确的地方改掉：
  1. 「计算本轮预算」不是独立一步，预算边调度边扣；
  2. 「选择 Prefill / Decode 任务」并不存在——调度器只让 num_computed_tokens 追上
     num_tokens_with_spec，没有阶段分支；
  3. 补上最实质的动作「抢占」：allocate_slots 拿不到 block 时踢 running 尾部请求。

输出：tmp-diagram/scheduler-step.drawio
"""
import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scheduler-step.drawio")

PAGE_W, PAGE_H = 1240, 1960

# ---------- 配色（沿用学习库既有图：墨色 / 蓝 / 绿 / 红 / 橙 / 紫）----------
INK = "#1E1E1E"
C_BLUE = "#2E6FDA"
C_GREEN = "#7CB342"
C_RED = "#E53935"
C_ORANGE = "#F39C12"
C_PURPLE = "#8E44AD"
GREEN_FILL = "#DFF0D8"
RED_FILL = "#FFF0EA"
ORANGE_FILL = "#FFF6E5"
BALL = "#4A78D0"

# ---------- 样式 ----------
def step_style(stroke=INK, fill="#FFFFFF", w=2, font=19, bold=True):
    return (f"rounded=0;whiteSpace=wrap;html=1;strokeWidth={w};fillColor={fill};strokeColor={stroke};"
            f"fontSize={font};fontStyle={1 if bold else 0};fontColor={INK};"
            f"verticalAlign=middle;align=center;spacingLeft=10;spacingRight=10;")

def green_style():
    return step_style(stroke=C_GREEN, fill=GREEN_FILL, w=3, font=21)

def zone_style(c, dashed=False):
    d = "dashed=1;dashPattern=8 5;" if dashed else ""
    return (f"rounded=1;arcSize=6;html=1;fillColor=none;strokeColor={c};strokeWidth=2;{d}"
            f"verticalAlign=top;align=left;fontSize=16;fontColor={c};fontStyle=1;"
            f"spacingLeft=14;spacingTop=8;")

SMALL = "rounded=0;whiteSpace=wrap;html=1;strokeWidth=1.5;fillColor=#FFFFFF;strokeColor=#7B8794;fontSize=15;fontColor=#333333;"
TEXT = "text;html=1;whiteSpace=wrap;align=left;verticalAlign=middle;"
CENTER_TEXT = "text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"

cells = []
_seq = [100]

def _id():
    _seq[0] += 1
    return "c%d" % _seq[0]

def vertex(x, y, w, h, txt, style):
    txt = txt.replace("\n", "<br>")
    i = _id()
    cells.append(
        f'        <mxCell id="{i}" value="{escape(txt)}" style="{style}" vertex="1" parent="1">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>\n'
        f'        </mxCell>')
    return i

def edge(src, dst, label="", c=INK, w=2, dashed=False, both=False,
         src_anchor=None, dst_anchor=None, label_pos=0.5):
    st = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;fontSize=14;fontColor=#555555;"
          "labelBackgroundColor=#FFFFFF;")
    st += f"strokeColor={c};strokeWidth={w};endArrow=block;endFill=1;"
    if dashed:
        st += "dashed=1;dashPattern=6 4;"
    if both:
        st += "startArrow=block;startFill=1;"
    if src_anchor:
        st += f"exitX={src_anchor[0]};exitY={src_anchor[1]};exitDx=0;exitDy=0;"
    if dst_anchor:
        st += f"entryX={dst_anchor[0]};entryY={dst_anchor[1]};entryDx=0;entryDy=0;"
    i = _id()
    cells.append(
        f'        <mxCell id="{i}" value="{escape(label)}" style="{st}" edge="1" parent="1" '
        f'source="{src}" target="{dst}">\n'
        f'          <mxGeometry x="{label_pos}" relative="1" as="geometry"/>\n'
        f'        </mxCell>')
    return i

def ball(x, y, n, color, d=34):
    st = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={color};strokeColor=#FFFFFF;"
          f"strokeWidth=2;fontColor=#FFFFFF;fontSize=14;fontStyle=1;")
    return vertex(x, y, d, d, n, st)

# ============================================================
# 标题
# ============================================================
vertex(60, 30, 1120, 44, "vLLM V1 一次调度：一个 EngineCore.step() 做了什么",
       "text;html=1;fontSize=27;fontStyle=1;fontColor=#1F4E79;align=left;verticalAlign=middle;")
vertex(60, 78, 1120, 30,
       "口径：vllm/v1/core/sched/scheduler.py + vllm/v1/engine/core.py（main 00b7847c80）",
       "text;html=1;fontSize=14;fontColor=#777777;align=left;verticalAlign=middle;")

CX = 560            # 主干中线
BW = 520            # 主干块宽
BX = CX - BW // 2   # 左边界

# ============================================================
# ① 接收请求
# ============================================================
v1 = vertex(BX, 140, BW, 84 + 26,
            "① 接收请求\nadd_request() → 进 waiting 队列，记录 prompt / 生成状态",
            step_style())
vertex(CX + BW // 2 + 24, 152, 300, 80,
       "每步开始时检查 has_requests()\n没有则这一步直接返回（不跑模型）",
       TEXT + "fontSize=13;fontColor=#777777;")

# ============================================================
# ② 本轮初始化：只在这里产生预算
# ============================================================
v2 = vertex(BX, 290, BW, 84,
            "② 本轮初始化（只此一次）\ntoken_budget = max_num_batched_tokens；input_budget 同理",
            step_style(stroke=C_ORANGE, fill=ORANGE_FILL))
vertex(CX + BW // 2 + 24, 300, 320, 84,
       "预算不是独立一步算出来的：\n从这里开始，每调度一个请求就扣一次",
       TEXT + "fontSize=13;fontColor=#8A6D1B;")

# ============================================================
# 相位框：Scheduler.schedule()
# ============================================================
zone = vertex(60, 525, 1120, 950, "Scheduler.schedule()", zone_style(C_BLUE))

# --- ③ RUNNING ---
v3 = vertex(BX, 580, BW, 100,
            "③ 先调度 RUNNING 请求\n本步要算的 token = tokens_with_spec + placeholders − computed",
            step_style(stroke=C_BLUE, fill="#F2F7FF"))
vertex(80, 580, 210, 100,
       "再用 token 预算 /\ninput 预算 / max_model_len\n三者取 min（chunked\nprefill 就是这里裁出来的）",
       TEXT + "fontSize=13;fontColor=#4A5A6E;")

# --- ④ allocate_slots / 抢占 ---
v4 = vertex(BX, 730, BW, 100,
            "④ allocate_slots()：给这批 token 分配 KV block\n拿不到 block 怎么办？",
            step_style(stroke=C_RED, fill=RED_FILL))
vertex(CX + BW // 2 + 24, 730, 340, 100,
       "显存是硬约束——token 预算还剩很多也可能卡在这里",
       TEXT + "fontSize=13;fontColor=#9B2C2C;")

# 抢占子框
sub = vertex(380, 865, 700, 236, "", zone_style(C_RED, dashed=True))
vertex(396, 870, 220, 26, "抢占 preemption", "text;html=1;fontSize=15;fontStyle=1;fontColor=%s;" % C_RED)
v5 = vertex(410, 910, 330, 66, "从 running 尾部踢出一个请求\n放回 waiting，释放它的 KV block", SMALL)
v6 = vertex(410, 994, 330, 66, "重新 allocate_slots()\n成功 → 继续；还是不行 → 再踢一个", SMALL)
vertex(780, 910, 280, 66,
       "_preempt_request()\nnum_preemptions += 1\nblock 归还池子", SMALL + "fillColor=#FFFFFF;")
vertex(780, 994, 280, 66,
       "下轮重新进 schedule()：\n前缀命中 → 只补增量\n未命中 → 整段重算", SMALL + "fillColor=#FFFFFF;")
edge(v5, v6, c=C_RED, w=1.8, src_anchor=(0.5, 1), dst_anchor=(0.5, 0))

# --- ⑤ WAITING ---
v7 = vertex(BX, 1145, BW, 100,
            "⑤ 本轮没有发生抢占，才调度 WAITING 请求\n（有抢占就整段跳过，防止刚腾的空间被抢回去）",
            step_style(stroke=C_GREEN, fill=GREEN_FILL))

# --- ⑥ 打包 ---
v8 = vertex(BX, 1295, BW, 100,
            "⑥ 打包 SchedulerOutput\nnum_scheduled_tokens / 各请求 block 表 / spec tokens",
            step_style())
vertex(CX + BW // 2 + 24, 1300, 340, 90,
       "preempted_req_ids、finished_req_ids\n也作为字段一起交给 worker",
       TEXT + "fontSize=13;fontColor=#4A5A6E;")

# 主干箭头
edge(v1, v2)
edge(v2, v3)
edge(v3, v4)
edge(v4, sub, label="失败", c=C_RED, w=1.8, src_anchor=(0.5, 1), dst_anchor=(0.5, 0))
edge(sub, v7, label="成功", c=C_GREEN, w=1.8, src_anchor=(0.5, 1), dst_anchor=(0.5, 0))
edge(v7, v8)

# ============================================================
# ⑦ GPU 执行（浅绿高亮，对应原图的绿色框）
# ============================================================
v9 = vertex(BX, 1555, BW, 96,
            "⑦ GPU 执行模型计算\nexecute_model(SchedulerOutput)",
            green_style())
edge(v8, v9, label="交给 worker", label_pos=0.35)

# ============================================================
# ⑧ 更新状态
# ============================================================
v10 = vertex(BX, 1705, BW, 100,
             "⑧ update_from_output()：追加 token / 判停止\n完成 → 释放 KV 与 encoder cache；未完成 → 留在 running",
             step_style(stroke=C_PURPLE, fill="#F6EFFA"))
edge(v9, v10)

# 回到 ③ 的循环箭头（右侧走线，避开说明文字）
v10_i = v10
st_loop = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=%s;strokeWidth=2;dashed=1;"
           "dashPattern=8 5;endArrow=block;endFill=1;fontSize=15;fontColor=%s;labelBackgroundColor=#FFFFFF;"
           % (C_BLUE, C_BLUE))
loop_pts = ('<Array as="points">'
            '<mxPoint x="1140" y="1755"/><mxPoint x="1140" y="630"/></Array>')
i = _id()
cells.append(
    f'        <mxCell id="{i}" value="下一轮 step：回到 ③（循环直到所有请求完成）" style="{st_loop}" edge="1" parent="1" '
    f'source="{v10}" target="{v3}">\n'
    f'          <mxGeometry relative="1" as="geometry">{loop_pts}</mxGeometry>\n'
    f'        </mxCell>')

# ============================================================
# 右侧：一句话纠偏 / 易混点
# ============================================================
vertex(60, 415, 1120, 96, "", zone_style(INK, dashed=True))
vertex(78, 424, 1086, 80,
       "这条链的本质 —— 调度器只做一件事：给「已算 token 数落后」的请求补预算，"
       "让 num_computed_tokens 追上 num_tokens_with_spec。<br>"
       "没有 prefill / decode 的分支判断（「在 prefill」只是差很多 token，「在 decode」只是差 1 个）；"
       "两个约束是 token 预算（能算多少）与 KV 显存（能放多少）。",
       TEXT + "fontSize=15;fontColor=#333333;")

vertex(BX, 1820, 900, 110,
       "读图注意：预算（②）不是独立一步，是边调度边扣；"
       "绿色高亮框（⑦）在调度器之外——调度器从不碰 GPU，只产出计划。",
       "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFBE6;strokeColor=#CFC286;"
       "fontColor=#5C4B00;fontSize=15;align=left;verticalAlign=middle;spacingLeft=16;spacingRight=16;")

# ============================================================
# 输出
# ============================================================
xml = (
    '<mxfile host="app.diagrams.net" type="device">\n'
    '  <diagram name="vllm 一次调度" id="vllm-scheduler-step">\n'
    f'    <mxGraphModel dx="{PAGE_W}" dy="1000" grid="0" gridSize="10" guides="1" tooltips="1" '
    f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{PAGE_W}" '
    f'pageHeight="{PAGE_H}" math="0" shadow="0">\n'
    '      <root>\n'
    '        <mxCell id="0"/>\n'
    '        <mxCell id="1" parent="0"/>\n'
    + "\n".join(cells) + "\n"
    + '      </root>\n'
    '    </mxGraphModel>\n'
    '  </diagram>\n'
    '</mxfile>\n'
)

ET.fromstring(xml)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(xml)
print("OK ->", OUT, len(xml), "bytes,", len(cells), "cells")
