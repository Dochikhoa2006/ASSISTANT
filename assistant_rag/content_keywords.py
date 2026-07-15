"""Deterministic keyword policy for generated-file requests."""

from __future__ import annotations


# These values intentionally mirror the user-supplied verb keyword policy.  Routing
# code treats them as exact words or phrases rather than arbitrary substrings.
FILE_CREATION_VERB_KEYWORDS: tuple[str, ...] = (
    "create", "generate", "make", "build", "produce", "prepare", "draft",
    "write", "compose", "construct", "develop", "design", "form", "assemble",
    "compile", "render", "export", "convert", "transform", "turn into",
    "put together", "set up", "lay out", "format", "organize", "structure",
    "author", "publish", "save as", "download as", "output as", "deliver as",
    "provide as", "give me", "send me", "prepare me", "make me", "create me",
    "generate me", "write me", "build me", "edit", "modify", "update",
    "revise", "rewrite", "rework", "improve", "enhance", "polish",
    "proofread", "correct", "fix", "reformat", "redesign", "restructure",
    "reorganize", "expand", "shorten", "summarize", "simplify", "translate",
    "merge", "combine", "split", "append", "add to", "remove from", "replace",
    "populate", "fill in", "complete", "finalize", "professionalize", "clean up",
    "touch up", "package", "attach", "share", "present", "print", "download",
    "save", "upload", "import", "extract", "analyze", "visualize", "chart",
    "graph", "calculate", "plan", "schedule", "track", "record", "document",
    "list", "tabulate", "map", "outline", "summarise", "proof", "refine",
    "adapt", "repurpose", "recreate", "rebuild", "regenerate",
)


# The supplied file list is partitioned by the only three artifact tools exposed by
# the composer.  Non-spreadsheet/non-presentation formats use the document tool,
# which currently materializes a PDF artifact.
DOCUMENT_FILE_KEYWORDS: tuple[str, ...] = (
    "file", "document", "artifact", "attachment", "downloadable file",
    "editable file", "template", "word", "word document", "microsoft word",
    "ms word", "doc", "docx", ".doc", ".docx", "text document",
    "editable document", "formatted document", "business document",
    "professional document", "office document", "letter", "business letter",
    "cover letter", "recommendation letter", "notice", "announcement", "memo",
    "memorandum", "report", "formal report", "research report",
    "technical report", "design report", "project report", "progress report",
    "annual report", "monthly report", "weekly report", "meeting minutes",
    "minutes of meeting", "agenda", "proposal", "business proposal",
    "project proposal", "research proposal", "statement", "personal statement",
    "policy", "procedure", "standard operating procedure", "sop", "manual",
    "guide", "handbook", "documentation", "specification",
    "requirements document", "system design document", "architecture document",
    "strategy memo", "legal memorandum", "investment memo", "executive brief",
    "executive summary", "white paper", "case study", "essay", "article",
    "assignment", "paper", "research paper", "thesis", "dissertation", "resume",
    "curriculum vitae", "cv", "job application", "contract", "agreement",
    "invoice", "quotation", "quote", "purchase order", "form", "questionnaire",
    "survey", "checklist", "pdf", ".pdf", "pdf file", "pdf document",
    "print-ready pdf", "downloadable pdf", "fillable pdf", "interactive pdf",
    "signed pdf", "portable document", "pdf report", "pdf guide", "pdf proposal",
    "pdf invoice", "pdf form", "pdf brochure", "pdf booklet", "pdf manual",
    "pdf portfolio", "pdf resume", "pdf presentation", "text file", "txt",
    ".txt", "plain text", "markdown", "md", ".md", "readme", "readme file",
    "markdown document", "html", ".html", "web page", "json", ".json", "yaml",
    "yml", ".yaml", ".yml", "xml", ".xml",
)


EXCEL_FILE_KEYWORDS: tuple[str, ...] = (
    "excel", "microsoft excel", "ms excel", "xlsx", "xls", ".xlsx", ".xls",
    "spreadsheet", "workbook", "worksheet", "sheet", "excel file", "excel sheet",
    "excel workbook", "google sheet", "google sheets", "csv", ".csv",
    "comma-separated values", "table", "data table", "tracker", "project tracker",
    "task tracker", "expense tracker", "budget tracker", "habit tracker",
    "inventory tracker", "attendance tracker", "sales tracker", "lead tracker",
    "pipeline tracker", "crm tracker", "financial model", "budget",
    "financial budget", "forecast", "financial forecast", "cash-flow forecast",
    "cash flow", "income statement", "balance sheet", "profit and loss", "p&l",
    "three-statement model", "three-statement forecast", "valuation model",
    "scenario model", "pricing model", "cost model", "calculator", "schedule",
    "operating calendar", "calendar", "timeline", "gantt chart", "roadmap",
    "resource plan", "capacity plan", "staffing plan", "shift schedule", "roster",
    "timesheet", "payroll sheet", "invoice sheet", "sales pipeline", "customer list",
    "contact list", "inventory", "stock list", "product catalog", "price list",
    "comparison table", "scorecard", "kpi scorecard", "dashboard",
    "analytics dashboard", "performance dashboard", "reporting dashboard",
    "pivot table", "chart", "graph", "dataset", "database export",
    "data entry form", "to-do list", "action register", "risk register", "issue log",
    "decision log", "meeting tracker", "project plan", "launch plan", "campaign plan",
)


POWERPOINT_FILE_KEYWORDS: tuple[str, ...] = (
    "powerpoint", "power point", "microsoft powerpoint", "ms powerpoint", "ppt", "pptx", ".ppt",
    ".pptx", "presentation", "slide presentation", "slide deck", "deck", "slides",
    "slideshow", "slide show", "pitch deck", "investor deck", "sales deck", "marketing deck",
    "training deck", "teaching slides", "lecture slides", "conference presentation",
    "seminar presentation", "webinar presentation", "project presentation",
    "project kickoff", "business review", "operating review",
    "quarterly business review", "qbr", "monthly business review",
    "executive presentation", "board presentation", "board deck",
    "management presentation", "strategy presentation", "roadmap presentation",
    "product presentation", "product demo deck", "company profile deck",
    "proposal presentation", "research presentation", "case study presentation",
    "market trends report", "market analysis presentation",
    "competitive analysis deck", "team alignment deck", "offsite presentation",
    "workshop deck", "training presentation", "lesson slides", "visual presentation",
    "storyboard", "speaker notes",
)


FILE_KEYWORDS: tuple[str, ...] = (
    *DOCUMENT_FILE_KEYWORDS,
    *EXCEL_FILE_KEYWORDS,
    *POWERPOINT_FILE_KEYWORDS,
)
