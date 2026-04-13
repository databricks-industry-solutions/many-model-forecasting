# MMF → Genie Code: Change Log

---

## SKILL.md

| Section | Original | Genie version | Why |
|---|---|---|---|
| Description (frontmatter) | "...using Databricks MCP tools and `AskUserQuestion`" | Removed | Genie Code doesn't use MCP tools or AskUserQuestion — those are Claude Code-specific |
| Overview | "Five **skills** walk you..." | "Five **steps** walk you..." | "Skills" is a specific concept in Genie Code — calling them steps avoids confusion |
| Workflow diagram | "submit parallel jobs" | "upload & trigger jobs" | More accurate: Genie Code uploads notebooks then triggers jobs |
| Prerequisites | "Databricks MCP server configured" | Removed → "Permissions to create clusters and jobs" | Genie Code runs natively inside Databricks — no external MCP server needed |
| Notebook templates | Direct `.ipynb` references | Points to `notebooks/` subfolder + upload note | Templates are bundled inside the skill folder so Genie Code can upload them to the workspace |
