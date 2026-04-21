"""Multi-agent dialogue orchestrator for medical education tutoring."""
import json
import re
from typing import Dict, Optional
from loguru import logger

from .base import LLMBackend, DialogueState, Message
from .prompts import (
    TUTOR_SYSTEM_PROMPT,
    PATIENT_SYSTEM_PROMPT,
    ASSESSMENT_SYSTEM_PROMPT,
    get_simulated_student_prompt,
)


def clean_output(text: str) -> str:
    """Remove garbage from LLM output: hashtags, repeated blocks, role bleeding."""
    # Remove hashtag spam (#MedEd #USMLE etc.)
    text = re.sub(r'#\w+', '', text)
    # Remove repeated phrases (3+ consecutive repeats of same 20+ char block)
    text = re.sub(r'(.{20,}?)\1{2,}', r'\1', text)
    # Remove lines that look like role instructions bleeding through
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line_lower = line.strip().lower()
        if any(skip in line_lower for skip in [
            'please respond as', 'you are the tutor', 'you are the student',
            'you are the patient', 'correct answer:', 'category:', 'step1',
            'step2', 'step3', 'meta_info', 'answer_idx'
        ]):
            continue
        clean_lines.append(line)
    text = '\n'.join(clean_lines).strip()
    # Truncate at reasonable length (max ~300 words per response)
    words = text.split()
    if len(words) > 300:
        text = ' '.join(words[:300])
    return text


class MultiAgentOrchestrator:
    """Orchestrates multi-agent tutoring dialogue: Tutor ↔ Student, with Patient and Assessment."""

    def __init__(self, llm: LLMBackend, case: Dict, student_config: Dict):
        self.llm = llm
        self.case = case
        self.student_config = student_config

    def run_dialogue(self, state: DialogueState) -> DialogueState:
        """Run a full multi-agent tutoring dialogue until completion."""
        case_desc = self.case["question"]
        patient_info = self._build_patient_context()
        student_prompt_base = get_simulated_student_prompt(self.student_config, case_desc)

        # Initial: Patient presents chief complaint
        initial_presentation = self._generate_patient_opening(patient_info)
        state.add_message("patient", initial_presentation)

        # Tutor opens with a question
        tutor_opening = self._generate_tutor_response(state, case_desc)
        state.add_message("tutor", tutor_opening)

        while not state.is_complete():
            # Student responds
            student_response = self._generate_student_response(
                state, student_prompt_base, case_desc
            )
            state.add_message("student", student_response)

            # Assessment agent evaluates
            assessment = self._assess_response(state, case_desc)
            state.assessments.append(assessment)

            # Check if student reached diagnosis
            if self._check_diagnosis(student_response, self.case.get("answer", "")):
                state.diagnosis_reached = True
                state.final_diagnosis = self._extract_diagnosis(student_response)
                # Tutor confirms
                confirm = self._generate_tutor_response(
                    state, case_desc, extra="The student seems to have reached the correct diagnosis. Confirm and summarize."
                )
                state.add_message("tutor", confirm)
                break

            if state.turn_count >= state.max_turns:
                break

            # Student may ask patient a question
            if self._student_wants_info(student_response):
                patient_reply = self._generate_patient_reply(state, patient_info)
                state.add_message("patient", patient_reply)

            # Tutor responds with scaffolding
            teaching_rec = assessment.get("teaching_recommendation", "")
            tutor_response = self._generate_tutor_response(
                state, case_desc, extra=teaching_rec
            )
            state.add_message("tutor", tutor_response)

        return state

    def _generate_patient_opening(self, patient_info: str) -> str:
        prompt = (
            "You are a patient arriving at the clinic. Briefly describe your main "
            "complaint in 2-3 sentences using everyday language.\n\n"
            f"Your medical details: {patient_info}\n\n"
            "IMPORTANT: Respond ONLY as the patient. Do NOT include any medical "
            "terminology, diagnoses, hashtags, or meta-commentary. Just describe "
            "your symptoms as a regular person would.\n\n"
            "Patient says:"
        )
        return clean_output(self.llm.generate(prompt))

    def _generate_tutor_response(
        self, state: DialogueState, case_desc: str, extra: str = ""
    ) -> str:
        history = state.get_dialogue_history()
        assessed = f"Turn {state.turn_count}"
        prompt = TUTOR_SYSTEM_PROMPT.format(
            case_description=case_desc, assessed_level=assessed
        )
        prompt += f"\n\nDialogue so far:\n{history}\n"
        if extra:
            prompt += f"\nInternal note: {extra}\n"
        prompt += (
            "\nIMPORTANT: Respond ONLY as the tutor. Write 2-4 sentences max. "
            "No hashtags, no meta-commentary, no role instructions.\n\n"
            "Tutor responds:"
        )
        return clean_output(self.llm.generate(prompt))

    def _generate_student_response(
        self, state: DialogueState, student_base_prompt: str, case_desc: str
    ) -> str:
        history = state.get_dialogue_history()
        prompt = (
            f"{student_base_prompt}\n\n"
            f"Dialogue so far:\n{history}\n\n"
            f"As this medical student, respond to the tutor's last message. "
            f"Show your reasoning process. Stay within your knowledge boundaries.\n\n"
            f"IMPORTANT: Write 2-5 sentences only. Respond ONLY as the student. "
            f"No hashtags, no role instructions, no meta-commentary.\n\n"
            f"Student responds:"
        )
        return clean_output(self.llm.generate(prompt))

    def _generate_patient_reply(self, state: DialogueState, patient_info: str) -> str:
        history = state.get_dialogue_history()
        last_student = ""
        for msg in reversed(state.messages):
            if msg.role == "student":
                last_student = msg.content
                break

        prompt = (
            f"You are the patient. A medical student asked you: \"{last_student[:200]}\"\n\n"
            f"Your medical details: {patient_info}\n\n"
            f"Answer the student's question naturally, as a real patient would. "
            f"Only reveal information that matches what they asked about.\n\n"
            f"IMPORTANT: Respond ONLY as the patient in 1-3 sentences. Use lay language. "
            f"NEVER mention diagnoses, medical terms, correct answers, or hashtags.\n\n"
            f"Patient says:"
        )
        return clean_output(self.llm.generate(prompt))

    def _assess_response(self, state: DialogueState, case_desc: str) -> Dict:
        last_student = ""
        for msg in reversed(state.messages):
            if msg.role == "student":
                last_student = msg.content
                break

        prompt = ASSESSMENT_SYSTEM_PROMPT.format(
            case_description=case_desc,
            student_level=self.student_config["label"],
            known_domains=", ".join(self.student_config["known_domains"]),
        )
        prompt += (
            f"\n\nStudent's response to evaluate:\n\"{last_student[:300]}\"\n\n"
            f"IMPORTANT: Output ONLY a single JSON object. No explanation, no text "
            f"before or after. Start with {{ and end with }}.\n\n"
            f"JSON:"
        )
        raw = self.llm.generate(prompt)
        # Extra cleaning for assessment
        raw = clean_output(raw)

        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {
            "blooms_level": "unknown",
            "coi_phase": "unknown",
            "reasoning_quality": 0,
            "misconceptions": [],
            "boundary_violations": [],
            "teaching_recommendation": "",
            "raw_assessment": raw,
        }

    def _check_diagnosis(self, student_text: str, correct_answer: str) -> bool:
        """Check if student mentioned the correct answer option's content."""
        if not correct_answer:
            return False
        # Get the actual answer text from options
        options = self.case.get("options", {})
        answer_text = options.get(correct_answer, correct_answer)
        # Check if key terms from the answer appear in student text
        # Use first 3 significant words (skip short words)
        answer_words = [w.lower() for w in answer_text.split() if len(w) > 4][:3]
        text_lower = student_text.lower()
        matches = sum(1 for w in answer_words if w in text_lower)
        return matches >= 2  # At least 2 key terms match

    def _extract_diagnosis(self, text: str) -> str:
        """Extract the diagnosis from student text."""
        # Look for common patterns
        for pattern in ["I think the diagnosis is", "diagnosis:", "I believe this is"]:
            if pattern.lower() in text.lower():
                idx = text.lower().index(pattern.lower())
                return text[idx : idx + 100].strip()
        return text[:100]

    def _student_wants_info(self, student_text: str) -> bool:
        """Detect if the student is asking the patient for information."""
        info_signals = [
            "can you tell me",
            "do you have",
            "when did",
            "how long",
            "does it hurt",
            "any history of",
            "are you taking",
            "family history",
            "I'd like to ask the patient",
            "let me examine",
            "physical exam",
            "lab results",
            "order",
        ]
        text_lower = student_text.lower()
        return any(signal in text_lower for signal in info_signals)

    def _build_patient_context(self) -> str:
        """Build patient information string from case data. NEVER include answer or metadata."""
        # Only include the clinical vignette, NOT the answer or category
        question = self.case.get("question", "")
        # Extract just the clinical scenario (before the "Which of the following" part)
        if "which of the following" in question.lower():
            idx = question.lower().index("which of the following")
            question = question[:idx].strip()
        return f"Clinical scenario: {question[:500]}"


class SingleAgentSimulator:
    """Single-agent baseline: one LLM simulates the entire interaction."""

    def __init__(self, llm: LLMBackend, case: Dict, student_config: Dict):
        self.llm = llm
        self.case = case
        self.student_config = student_config

    def run_dialogue(self, state: DialogueState) -> DialogueState:
        from .prompts import SINGLE_AGENT_PROMPT

        prompt = SINGLE_AGENT_PROMPT.format(
            student_level=self.student_config["label"],
            known_domains=", ".join(self.student_config["known_domains"]),
            unknown_domains=", ".join(self.student_config["unknown_domains"]),
            case_description=self.case["question"],
        )
        prompt += "\n\nGenerate a complete tutoring dialogue (15-20 turns):\n"

        full_dialogue = self.llm.generate(prompt)
        state.add_message("unified", full_dialogue)

        # Parse dialogue into individual messages for analysis
        for line in full_dialogue.split("\n"):
            line = line.strip()
            if line.startswith("[PATIENT]:"):
                state.add_message("patient", line[10:].strip(), {"parsed": True})
            elif line.startswith("[STUDENT]:"):
                state.add_message("student", line[10:].strip(), {"parsed": True})
            elif line.startswith("[TUTOR]:"):
                state.add_message("tutor", line[8:].strip(), {"parsed": True})

        return state


class DirectQASimulator:
    """Direct QA baseline: student asks, LLM answers directly."""

    def __init__(self, llm: LLMBackend, case: Dict, student_config: Dict):
        self.llm = llm
        self.case = case
        self.student_config = student_config

    def run_dialogue(self, state: DialogueState) -> DialogueState:
        from .prompts import DIRECT_QA_PROMPT, get_simulated_student_prompt

        case_desc = self.case["question"]
        student_prompt = get_simulated_student_prompt(self.student_config, case_desc)
        qa_prompt = DIRECT_QA_PROMPT.format(case_description=case_desc)

        # Student asks initial question
        student_q = self.llm.generate(
            f"{student_prompt}\n\nYou're given a clinical case. Ask your first question.\n"
            f"Case: {case_desc}\n\nStudent asks:"
        )
        state.add_message("student", student_q)

        for _ in range(min(10, state.max_turns // 2)):
            # QA system answers directly
            qa_answer = self.llm.generate(
                f"{qa_prompt}\n\nStudent's question: {student_q}\n\nDirect answer:"
            )
            state.add_message("qa", qa_answer)

            # Student follows up
            history = state.get_dialogue_history()
            student_q = self.llm.generate(
                f"{student_prompt}\n\nDialogue:\n{history}\n\n"
                f"Ask a follow-up question or state your diagnosis.\nStudent:"
            )
            state.add_message("student", student_q)

            if self._check_diagnosis(student_q, self.case.get("answer", "")):
                state.diagnosis_reached = True
                break

        return state

    def _check_diagnosis(self, text: str, answer: str) -> bool:
        if not answer:
            return False
        return answer.lower()[:30] in text.lower()
