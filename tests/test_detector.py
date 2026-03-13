from anonymizer.pii.detector import (
    DetectionResult,
    OcrToken,
    _group_into_lines,
    _scan_line,
    detect,
)


def tok(text: str, x1: int, y1: int, x2: int, y2: int) -> OcrToken:
    return OcrToken(text=text, bbox=(x1, y1, x2, y2))


class TestDetectReturnType:
    def test_returns_detection_result(self):
        assert isinstance(detect([]), DetectionResult)

    def test_redact_boxes_is_list(self):
        assert isinstance(detect([]).redact_boxes, list)

    def test_empty_tokens_empty_boxes(self):
        assert detect([]).redact_boxes == []

    def test_no_labels_empty_boxes(self):
        tokens = [tok("Hello", 0, 0, 50, 20), tok("World", 60, 0, 120, 20)]
        assert detect(tokens).redact_boxes == []


class TestNameLabelDetection:
    def test_name_abi_flags_value(self):
        tokens = [tok("Name", 0, 0, 50, 20), tok("John Smith", 60, 0, 180, 20)]
        assert (60, 0, 180, 20) in detect(tokens).redact_boxes

    def test_name_label_itself_not_in_results(self):
        tokens = [tok("Name", 0, 0, 50, 20), tok("John Smith", 60, 0, 180, 20)]
        assert (0, 0, 50, 20) not in detect(tokens).redact_boxes

    def test_name_colon_neuro_touch_flags_value(self):
        tokens = [tok("NAME:", 0, 0, 50, 20), tok("Sridhar Singh", 60, 0, 200, 20)]
        assert (60, 0, 200, 20) in detect(tokens).redact_boxes

    def test_subject_name_plantar_scan_flags_value(self):
        tokens = [
            tok("Subject name", 0, 0, 100, 20),
            tok("Sridhar Singh M G", 110, 0, 300, 20),
        ]
        assert (110, 0, 300, 20) in detect(tokens).redact_boxes

    def test_name_label_multi_token_name(self):
        tokens = [
            tok("NAME:", 0, 0, 50, 20),
            tok("Sridhar", 60, 0, 120, 20),
            tok("Singh", 130, 0, 180, 20),
            tok("M", 190, 0, 210, 20),
            tok("G", 220, 0, 240, 20),
        ]
        result = detect(tokens)
        assert (60, 0, 120, 20) in result.redact_boxes
        assert (130, 0, 180, 20) in result.redact_boxes
        assert (190, 0, 210, 20) in result.redact_boxes
        assert (220, 0, 240, 20) in result.redact_boxes

    def test_name_value_stops_at_stop_word(self):
        tokens = [
            tok("Name", 0, 0, 50, 20),
            tok("John Smith", 60, 0, 180, 20),
            tok("GENDER", 190, 0, 260, 20),
            tok("Male", 270, 0, 320, 20),
        ]
        result = detect(tokens)
        assert (60, 0, 180, 20) in result.redact_boxes
        assert (190, 0, 260, 20) not in result.redact_boxes
        assert (270, 0, 320, 20) not in result.redact_boxes

    def test_name_value_stops_at_another_label(self):
        tokens = [
            tok("Name", 0, 0, 50, 20),
            tok("John", 60, 0, 110, 20),
            tok("PATIENT ID:", 120, 0, 230, 20),
        ]
        result = detect(tokens)
        assert (60, 0, 110, 20) in result.redact_boxes
        assert (120, 0, 230, 20) not in result.redact_boxes

    def test_name_label_no_following_token_no_crash(self):
        assert detect([tok("Name", 0, 0, 50, 20)]).redact_boxes == []

    def test_lowercase_name_label(self):
        tokens = [tok("name", 0, 0, 50, 20), tok("Jane Doe", 60, 0, 160, 20)]
        assert (60, 0, 160, 20) in detect(tokens).redact_boxes


class TestStrongIdLabelDetection:
    def test_patient_id_colon_flags_value(self):
        tokens = [
            tok("PATIENT ID:", 0, 0, 100, 20),
            tok("KIER175326", 110, 0, 200, 20),
        ]
        assert (110, 0, 200, 20) in detect(tokens).redact_boxes

    def test_strong_id_label_not_in_results(self):
        tokens = [
            tok("PATIENT ID:", 0, 0, 100, 20),
            tok("KIER175326", 110, 0, 200, 20),
        ]
        assert (0, 0, 100, 20) not in detect(tokens).redact_boxes

    def test_strong_id_flags_any_value_format(self):
        tokens = [
            tok("PATIENT ID:", 0, 0, 100, 20),
            tok("99999", 110, 0, 160, 20),
        ]
        assert (110, 0, 160, 20) in detect(tokens).redact_boxes

    def test_strong_id_label_no_following_token_no_crash(self):
        assert detect([tok("PATIENT ID:", 0, 0, 100, 20)]).redact_boxes == []


class TestWeakIdLabelDetection:
    def test_id_with_kier_value_flagged(self):
        tokens = [tok("ID", 0, 0, 30, 20), tok("KIER34498", 40, 0, 120, 20)]
        assert (40, 0, 120, 20) in detect(tokens).redact_boxes

    def test_id_label_itself_not_in_results(self):
        tokens = [tok("ID", 0, 0, 30, 20), tok("KIER34498", 40, 0, 120, 20)]
        assert (0, 0, 30, 20) not in detect(tokens).redact_boxes

    def test_id_with_non_kier_value_not_flagged(self):
        tokens = [tok("ID", 0, 0, 30, 20), tok("12345", 40, 0, 100, 20)]
        assert (40, 0, 100, 20) not in detect(tokens).redact_boxes

    def test_id_colon_with_kier_value_flagged(self):
        tokens = [tok("id:", 0, 0, 30, 20), tok("KIER1234", 40, 0, 120, 20)]
        assert (40, 0, 120, 20) in detect(tokens).redact_boxes

    def test_id_no_following_token_no_crash(self):
        assert detect([tok("ID", 0, 0, 30, 20)]).redact_boxes == []


class TestLineGrouping:
    def test_group_empty(self):
        assert _group_into_lines([]) == []

    def test_group_single_token(self):
        t = tok("Hello", 0, 0, 50, 20)
        lines = _group_into_lines([t])
        assert len(lines) == 1
        assert lines[0] == [t]

    def test_group_sorts_tokens_by_x(self):
        tokens = [tok("B", 100, 0, 150, 20), tok("A", 0, 0, 90, 20)]
        lines = _group_into_lines(tokens)
        assert lines[0][0].text == "A"
        assert lines[0][1].text == "B"

    def test_group_separates_distant_y_tokens(self):
        tokens = [tok("L1", 0, 0, 50, 20), tok("L2", 0, 100, 50, 120)]
        lines = _group_into_lines(tokens)
        assert len(lines) == 2

    def test_group_merges_tokens_within_tolerance(self):
        tokens = [tok("A", 0, 0, 50, 20), tok("B", 60, 5, 120, 25)]
        lines = _group_into_lines(tokens)
        assert len(lines) == 1
        assert len(lines[0]) == 2

    def test_group_at_exact_tolerance_boundary_merged(self):
        tokens = [tok("A", 0, 0, 50, 20), tok("B", 60, 10, 120, 30)]
        lines = _group_into_lines(tokens)
        assert len(lines) == 1

    def test_group_just_beyond_tolerance_splits(self):
        tokens = [tok("A", 0, 0, 50, 20), tok("B", 60, 11, 120, 31)]
        lines = _group_into_lines(tokens)
        assert len(lines) == 2

    def test_group_three_distinct_lines(self):
        tokens = [
            tok("R1", 0, 0, 50, 20),
            tok("R2", 0, 50, 50, 70),
            tok("R3", 0, 100, 50, 120),
        ]
        assert len(_group_into_lines(tokens)) == 3


class TestCrossLineIsolation:
    def test_name_label_does_not_reach_next_line(self):
        tokens = [
            tok("Name", 0, 0, 50, 20),
            tok("John", 0, 100, 80, 120),
        ]
        assert detect(tokens).redact_boxes == []

    def test_value_on_same_line_flagged(self):
        tokens = [
            tok("Name", 0, 0, 50, 20),
            tok("John Smith", 60, 5, 180, 25),
        ]
        assert (60, 5, 180, 25) in detect(tokens).redact_boxes


class TestMultipleLabels:
    def test_name_and_strong_id_both_detected(self):
        tokens = [
            tok("Name", 0, 0, 50, 20),
            tok("John Smith", 60, 0, 180, 20),
            tok("PATIENT ID:", 0, 30, 100, 50),
            tok("KIER1234", 110, 30, 200, 50),
        ]
        result = detect(tokens)
        assert (60, 0, 180, 20) in result.redact_boxes
        assert (110, 30, 200, 50) in result.redact_boxes

    def test_name_and_weak_id_both_detected(self):
        tokens = [
            tok("Name", 0, 0, 50, 20),
            tok("Jane Doe", 60, 0, 160, 20),
            tok("ID", 0, 30, 30, 50),
            tok("KIER5678", 40, 30, 130, 50),
        ]
        result = detect(tokens)
        assert (60, 0, 160, 20) in result.redact_boxes
        assert (40, 30, 130, 50) in result.redact_boxes
