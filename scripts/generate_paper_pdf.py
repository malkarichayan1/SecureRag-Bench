"""Generate the review-ready offline paper draft PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "securerag_bench_offline_draft.pdf"


def section(title: str, text: str, styles: dict) -> list:
    return [Paragraph(title, styles["Heading2"]), Paragraph(text, styles["BodyText"]), Spacer(1, 8)]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=1, fontSize=16, leading=19))
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 12
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
    )
    story = [
        Paragraph("Trigger Fragments and Architectural LLM Defense", styles["TitleCenter"]),
        Spacer(1, 4),
        Paragraph("SecureRAG-Bench: Offline Deterministic Research Draft", styles["Italic"]),
        Spacer(1, 12),
    ]
    story += section("Abstract", "Indirect prompt injection turns retrieved text into a control channel for tool-using language-model agents. We present SecureRAG-Bench, an offline deterministic reference implementation with a privileged planner, quarantined parser, and restricted-AST reference monitor. In 15 benign tasks and seven attack paths, the full monitor retained 100% utility and reduced attack success rate from 100% without monitoring to 0%; policy-only enforcement left 42.86% attack success. A ten-seed cross-entropy trigger study further showed that optimized fragments can move malicious content into the retrieval top five, motivating architectural rather than prompt-only defenses.", styles)
    story += section("1. Threat Model and Design", "The attacker controls retrieved text but not the original objective, privileged planner, quarantine interface, or reference monitor. The privileged planner receives only the user objective. The quarantined parser receives only XML-delimited untrusted content and has no tools. The interpreter validates a restricted Python AST, tracks provenance and allowed readers, and checks task alignment, source authorization, action alignment, and data isolation before each tool call.", styles)
    story += section("2. Related Work and Contribution", "SecureRAG-Bench is a reproducible implementation and evaluation artifact inspired by the CaMeL capability-based architecture [4], not a claim to introduce that architecture. AgentDojo [1] and InjecAgent [2] provide complementary evaluation settings. Task Shield [5] evaluates task alignment; this work treats it as complementary to provenance enforcement because an otherwise authorized action can still carry tool-sourced data.", styles)
    story += section("3. Experimental Setup", "All experiments use deterministic mock components, simulated external capabilities, and no paid API. The same 15 benign workspace/banking tasks and seven retrieved-content attack paths are executed for all variants. The comparison includes no monitoring, XML delimiters without enforcement, policy-only enforcement, and the full monitor.", styles)
    story.append(Paragraph("4. Reference-Monitor Ablation", styles["Heading2"]))
    table = Table([
        ["Variant", "Benign", "Attacks", "Utility", "ASR", "Halt rate"],
        ["No monitor", "15", "7", "100%", "100%", "0%"],
        ["XML only", "15", "7", "100%", "100%", "0%"],
        ["Policy only", "15", "7", "100%", "42.86%", "57.14%"],
        ["Full monitor", "15", "7", "100%", "0%", "100%"],
    ], colWidths=[1.35*inch, .65*inch, .65*inch, .7*inch, .55*inch, .85*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#7F8C8D")),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#F4F8FB")),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story += [table, Spacer(1, 8), Paragraph("XML labels alone did not block the tested flows. Policy-only enforcement blocked four paths, while provenance enforcement blocked the remaining authorized-action taint paths. Benign utility remained unchanged in this deterministic suite.", styles["BodyText"]), Spacer(1, 8)]
    story += section("5. Trigger-Fragment Study", "We apply the Cross-Entropy Method to optimize a ten-token prefix for cosine similarity to a benign query over 30 iterations with 5,000 samples per iteration. Across seeds 1 through 10, the malicious document entered the top five in all runs. Mean best fitness was 0.906141 with population standard deviation 0.001968. This establishes retrieval manipulation under the deterministic setup; it does not establish a successful external action because the reference monitor still rejects tainted external sends.", styles)
    story += section("6. Offline Payload-Transfer Study", "The full monitor was also evaluated against 2,108 public InjecAgent base and enhanced records through a generic offline payload-transfer adapter. Transfer ASR was 0% and policy halts were 100%. This is not an official InjecAgent score because source tools are mapped to generic retrieval and a simulated external email action.", styles)
    story += section("7. Limitations", "This is an offline prototype, not an official AgentDojo or InjecAgent score. The suite and external capabilities are simulated, model components are deterministic mocks, and no real-model prompt-following claim is made. Claims are limited to the implemented restricted-AST architecture and saved offline artifacts.", styles)
    story += section("8. Reproducibility", "The controlled ablation is saved in artifacts/fast_urtc_controlled_evaluation.json, the ten-seed CEM output is artifacts/cem_extended_10.json, and the transfer study is artifacts/injecagent_offline_study.json. Run the commands in artifacts/README.md and python -m pytest -q from the repository root to verify the package.", styles)
    story += section("References", "[1] E. Debenedetti et al., AgentDojo, NeurIPS Datasets and Benchmarks Track, 2024. [2] Q. Zhan et al., InjecAgent, Findings of ACL, 2024. [3] R. Y. Rubinstein and D. P. Kroese, The Cross-Entropy Method, Springer, 2004. [4] E. Debenedetti et al., Defeating Prompt Injections by Design, 2025. [5] F. Jia et al., The Task Shield, ACL 2025.", styles)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
