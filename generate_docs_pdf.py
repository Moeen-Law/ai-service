#!/usr/bin/env python3
"""Generate PDF documentation for AI Orchestration Service."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)
from reportlab.lib import colors

# Create PDF
pdf_path = "docs/AI_Orchestration_Service_Documentation.pdf"
doc = SimpleDocTemplate(
    pdf_path, pagesize=A4, topMargin=0.75 * inch, bottomMargin=0.75 * inch
)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Title"],
    fontSize=24,
    spaceAfter=30,
    textColor=HexColor("#1a365d"),
)
heading_style = ParagraphStyle(
    "CustomHeading",
    parent=styles["Heading1"],
    fontSize=16,
    spaceBefore=20,
    spaceAfter=10,
    textColor=HexColor("#2c5282"),
)
subheading_style = ParagraphStyle(
    "CustomSubHeading",
    parent=styles["Heading2"],
    fontSize=12,
    spaceBefore=15,
    spaceAfter=8,
    textColor=HexColor("#4a5568"),
)
body_style = ParagraphStyle(
    "CustomBody", parent=styles["Normal"], fontSize=10, spaceAfter=8, leading=14
)
code_style = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=9,
    backColor=HexColor("#f7fafc"),
    borderColor=HexColor("#e2e8f0"),
    borderWidth=1,
    borderPadding=5,
    leftIndent=10,
    spaceAfter=10,
)

# Build content
content = []

# Title
content.append(Paragraph("AI Orchestration Service", title_style))
content.append(
    Paragraph("FastAPI Backend Implementation Documentation", styles["Heading2"])
)
content.append(Spacer(1, 20))

# Overview
content.append(Paragraph("1. Project Overview", heading_style))
content.append(
    Paragraph(
        "A FastAPI-based AI Orchestration Backend for a legal-tech microservices system. "
        "The service handles request routing, validation, and workflow orchestration without "
        "implementing AI logic directly. Built using Clean Architecture and Hexagonal patterns.",
        body_style,
    )
)

# Tech Stack
content.append(Paragraph("2. Technology Stack", heading_style))
tech_data = [
    ["Component", "Technology"],
    ["Framework", "FastAPI 0.109+"],
    ["Validation", "Pydantic 2.6+"],
    ["Configuration", "pydantic-settings 2.1+"],
    ["Testing", "pytest 9.0+, pytest-asyncio, httpx"],
    ["Logging", "Structured JSON with request ID"],
    ["Python", "3.13+"],
]
tech_table = Table(tech_data, colWidths=[2 * inch, 3 * inch])
tech_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f7fafc")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]
    )
)
content.append(tech_table)
content.append(Spacer(1, 15))

# Architecture
content.append(Paragraph("3. Architecture", heading_style))
content.append(
    Paragraph(
        "The project follows Clean Architecture with 4 layers and Hexagonal (Ports & Adapters) pattern:",
        body_style,
    )
)

arch_data = [
    ["Layer", "Purpose", "Location"],
    ["API", "HTTP endpoints, schemas, middleware", "app/api/"],
    ["Core", "Domain models, orchestrator, validators, workflows", "app/core/"],
    ["Interfaces", "Abstract ports for external services", "app/interfaces/"],
    ["Infrastructure", "Adapters, config, logging", "app/infrastructure/"],
]
arch_table = Table(arch_data, colWidths=[1.3 * inch, 2.7 * inch, 1.5 * inch])
arch_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]
    )
)
content.append(arch_table)
content.append(Spacer(1, 15))

# Build Steps
content.append(Paragraph("4. Implementation Steps", heading_style))

steps_data = [
    ["#", "Step", "Description", "Status"],
    ["1", "FastAPI Skeleton", "Project structure, main app, health endpoints", "✓"],
    ["2", "API Schemas", "Pydantic request/response models", "✓"],
    ["3", "Domain Models", "Entities, enums, value objects", "✓"],
    ["4", "Task Classification", "Explicit task type routing", "✓"],
    ["5", "Business Validation", "Payload validation per task type", "✓"],
    ["6", "Task Orchestrator", "Central coordination component", "✓"],
    ["7", "Workflow Registry", "Dynamic workflow management", "✓"],
    ["8", "Workflow Interface", "Abstract base for workflows", "✓"],
    ["9", "Stubbed Workflows", "5 workflow implementations", "✓"],
    ["10", "AI Service Interfaces", "LLM, RAG, Prompt ports", "✓"],
    ["11", "Infrastructure Adapters", "Stub service implementations", "✓"],
    ["12", "Structured Logging", "JSON logging with request ID", "✓"],
    ["13", "Testing Strategy", "Unit + integration tests", "✓"],
]
steps_table = Table(
    steps_data, colWidths=[0.4 * inch, 1.5 * inch, 2.8 * inch, 0.6 * inch]
)
steps_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (3, 0), (3, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("TEXTCOLOR", (3, 1), (3, -1), HexColor("#38a169")),
        ]
    )
)
content.append(steps_table)
content.append(Spacer(1, 15))

# Task Types
content.append(Paragraph("5. Supported Task Types", heading_style))

task_data = [
    ["Task Type", "Description"],
    ["LEGAL_CHAT", "Interactive legal Q&A conversations"],
    ["DOCUMENT_GENERATION", "Generate legal documents from templates"],
    ["CONTRACT_ANALYSIS", "Analyze contract clauses and terms"],
    ["CONTRACT_REFRAMING", "Rewrite clauses from different perspectives"],
    ["CASE_EVALUATION", "Evaluate legal cases and provide assessments"],
]
task_table = Table(task_data, colWidths=[2 * inch, 3.5 * inch])
task_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]
    )
)
content.append(task_table)
content.append(Spacer(1, 15))

# Jurisdictions
content.append(Paragraph("6. Supported Jurisdictions", heading_style))
content.append(Paragraph("• Egypt (EG) - Arabic and English", body_style))
content.append(Paragraph("• UAE (AE) - Arabic and English", body_style))
content.append(Spacer(1, 10))

# AI Service Interfaces
content.append(Paragraph("7. AI Service Interfaces (Ports)", heading_style))

interface_data = [
    ["Interface", "Purpose", "Key Methods"],
    [
        "LLMServiceInterface",
        "Language model operations",
        "generate(), generate_with_context()",
    ],
    [
        "RAGServiceInterface",
        "Vector search & retrieval",
        "retrieve(), search_similar()",
    ],
    [
        "PromptServiceInterface",
        "Template management",
        "get_template(), assemble_prompt()",
    ],
]
interface_table = Table(interface_data, colWidths=[1.5 * inch, 1.8 * inch, 2.2 * inch])
interface_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]
    )
)
content.append(interface_table)
content.append(Spacer(1, 15))

# Testing
content.append(Paragraph("8. Test Coverage", heading_style))

test_data = [
    ["Test Suite", "Count", "Scope"],
    ["Unit: TaskClassifier", "7", "Task type classification logic"],
    ["Unit: BusinessValidator", "13", "Payload validation rules"],
    ["Unit: WorkflowRegistry", "7", "Workflow registration/retrieval"],
    ["Integration: API", "14", "Full request/response cycle"],
    ["Total", "41", "All tests passing ✓"],
]
test_table = Table(test_data, colWidths=[1.8 * inch, 0.7 * inch, 3 * inch])
test_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("BACKGROUND", (0, -1), (-1, -1), HexColor("#e6fffa")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]
    )
)
content.append(test_table)
content.append(Spacer(1, 15))

# API Endpoints
content.append(Paragraph("9. API Endpoints", heading_style))

api_data = [
    ["Method", "Endpoint", "Description"],
    ["GET", "/health", "Basic health check"],
    ["GET", "/ready", "Readiness probe"],
    ["POST", "/api/v1/tasks", "Execute AI task"],
]
api_table = Table(api_data, colWidths=[0.8 * inch, 1.8 * inch, 2.9 * inch])
api_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2c5282")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [HexColor("#ffffff"), HexColor("#f7fafc")],
            ),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]
    )
)
content.append(api_table)
content.append(Spacer(1, 15))

# Run Instructions
content.append(Paragraph("10. Running the Service", heading_style))
content.append(Paragraph("Start the development server:", body_style))
content.append(Paragraph("uvicorn app.api.main:app --reload", code_style))
content.append(Paragraph("Run tests:", body_style))
content.append(Paragraph("pytest tests/ -v", code_style))

# Build PDF
doc.build(content)
print(f"PDF generated: {pdf_path}")
