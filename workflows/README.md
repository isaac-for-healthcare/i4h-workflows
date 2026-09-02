# Workflow Layout and Specialty Catalog

This page describes the workflow layout and specialty catalog.

A Scene defines the simulated world. A Task defines reusable behavior. A Workflow selects one Scene and connects it to the Tasks needed to complete a goal.

## Layout

Workflow definitions are grouped by clinical robotics specialty:

```text
workflows/
├── i4h_workflows/
│   ├── laparoscopic-robotics/
│   ├── ultrasound-robotics/
│   ├── endoluminal-robotics/
│   └── hospital-automation-robotics/
└── i4h_workflow_modes/
```

| Specialty | Workflows |
| --- | --- |
| [Laparoscopic robotics](i4h_workflows/laparoscopic-robotics/README.md) | `surgical_lift_block`, `surgical_lift_needle`, `surgical_lift_needle_organs`, `surgical_reach_dual_psm`, `surgical_reach_psm`, `surgical_reach_star` |
| [Ultrasound robotics](i4h_workflows/ultrasound-robotics/README.md) | `ultrasound_liver_scan`, `ultrasound_probe_reach` |
| [Endoluminal robotics](i4h_workflows/endoluminal-robotics/README.md) | `endoluminal_navigation` |
| [Hospital automation robotics](i4h_workflows/hospital-automation-robotics/README.md) | `assemble_trocar`, `locomanip_push_cart`, `locomanip_tray_pick_and_place`, `scissor_pick_and_place` |

The Python filename is the public workflow ID and must be unique across all specialties. Specialty folders organize the source only. Runtime commands use the workflow ID without the specialty path:

```bash
./run.sh surgical_reach_psm --rule-based
```

Standard run-mode builders shared by multiple Workflows live in [`i4h_workflow_modes/`](i4h_workflow_modes/README.md).
