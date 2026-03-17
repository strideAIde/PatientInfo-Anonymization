import pytest

from anonymizer.pii.patterns import (
    is_kier_id,
    is_name_label,
    is_name_stop_word,
    is_strong_id_label,
    is_weak_id_label,
    looks_like_kier_id,
    looks_like_patient_name,
)


class TestIsNameLabel:
    def test_subject_name_plantar_scan(self):
        assert is_name_label("Subject name") is True

    def test_subject_name_double_space(self):
        assert is_name_label("Subject  name") is True

    def test_name_colon_neuro_touch(self):
        assert is_name_label("NAME:") is True

    def test_name_no_colon_abi(self):
        assert is_name_label("Name") is True

    def test_lowercase(self):
        assert is_name_label("name:") is True

    def test_uppercase(self):
        assert is_name_label("NAME") is True

    def test_strips_surrounding_whitespace(self):
        assert is_name_label("  Name  ") is True

    def test_patient_id_is_not_name_label(self):
        assert is_name_label("PATIENT ID:") is False

    def test_bare_id_is_not_name_label(self):
        assert is_name_label("ID") is False

    def test_subject_alone_is_not_name_label(self):
        assert is_name_label("Subject") is False

    def test_random_text_is_not_name_label(self):
        assert is_name_label("John Smith") is False

    def test_empty_string_is_not_name_label(self):
        assert is_name_label("") is False


class TestIsStrongIdLabel:
    def test_patient_id_colon(self):
        assert is_strong_id_label("PATIENT ID:") is True

    def test_patient_id_no_colon(self):
        assert is_strong_id_label("PATIENT ID") is True

    def test_lowercase(self):
        assert is_strong_id_label("patient id:") is True

    def test_mixed_case(self):
        assert is_strong_id_label("Patient Id:") is True

    def test_strips_whitespace(self):
        assert is_strong_id_label("  PATIENT ID:  ") is True

    def test_bare_id_is_not_strong(self):
        assert is_strong_id_label("ID") is False

    def test_id_colon_is_not_strong(self):
        assert is_strong_id_label("ID:") is False

    def test_name_label_is_not_strong_id(self):
        assert is_strong_id_label("NAME:") is False

    def test_empty_is_not_strong_id(self):
        assert is_strong_id_label("") is False


class TestIsWeakIdLabel:
    def test_bare_id(self):
        assert is_weak_id_label("ID") is True

    def test_id_colon(self):
        assert is_weak_id_label("id:") is True

    def test_lowercase_id(self):
        assert is_weak_id_label("id") is True

    def test_strips_whitespace(self):
        assert is_weak_id_label("  ID  ") is True

    def test_patient_id_is_not_weak(self):
        assert is_weak_id_label("PATIENT ID:") is False

    def test_name_is_not_weak_id(self):
        assert is_weak_id_label("Name") is False

    def test_empty_is_not_weak_id(self):
        assert is_weak_id_label("") is False


class TestIsKierId:
    def test_kier_6_digits(self):
        assert is_kier_id("KIER175326") is True

    def test_kier_5_digits_with_space(self):
        assert is_kier_id("KIER 34498") is True

    def test_kier_4_digits_min(self):
        assert is_kier_id("KIER1234") is True

    def test_kier_8_digits_max(self):
        assert is_kier_id("KIER12345678") is True

    def test_lowercase_kier(self):
        assert is_kier_id("kier5678") is True

    def test_mixed_case_kier(self):
        assert is_kier_id("Kier9012") is True

    def test_strips_whitespace(self):
        assert is_kier_id("  KIER1234  ") is True

    def test_too_few_digits_rejected(self):
        assert is_kier_id("KIER123") is False

    def test_too_many_digits_rejected(self):
        assert is_kier_id("KIER123456789") is False

    def test_no_prefix_rejected(self):
        assert is_kier_id("175326") is False

    def test_wrong_prefix_rejected(self):
        assert is_kier_id("PATIENT175326") is False

    def test_empty_rejected(self):
        assert is_kier_id("") is False

    def test_kier_no_digits_rejected(self):
        assert is_kier_id("KIER") is False


class TestIsNameStopWord:
    def test_kier_is_stop_word(self):
        assert is_name_stop_word("KIER") is True

    def test_patient_is_stop_word(self):
        assert is_name_stop_word("PATIENT") is True

    def test_gender_is_stop_word(self):
        assert is_name_stop_word("GENDER") is True

    def test_dob_is_stop_word(self):
        assert is_name_stop_word("DOB") is True

    def test_lowercase_match(self):
        assert is_name_stop_word("kier") is True

    def test_mixed_case_match(self):
        assert is_name_stop_word("Patient") is True

    def test_strips_whitespace(self):
        assert is_name_stop_word("  KIER  ") is True

    def test_real_name_is_not_stop_word(self):
        assert is_name_stop_word("John Smith") is False

    def test_empty_is_not_stop_word(self):
        assert is_name_stop_word("") is False

    def test_partial_match_not_stop_word(self):
        assert is_name_stop_word("KIERKEGAARD") is False


class TestLooksLikePatientName:
    def test_two_word_name(self):
        assert looks_like_patient_name("REDDY M") is True

    def test_three_word_name(self):
        assert looks_like_patient_name("JAYARAMA REDDY M") is True

    def test_four_word_name(self):
        assert looks_like_patient_name("SRIDHAR SINGH M G") is True

    def test_single_word_is_not_name(self):
        assert looks_like_patient_name("SRIDHAR") is False

    def test_six_words_is_not_name(self):
        assert looks_like_patient_name("A B C D E F") is False

    def test_mixed_case_is_not_name(self):
        assert looks_like_patient_name("Sridhar Singh") is False

    def test_preposition_of_disqualifies(self):
        assert looks_like_patient_name("KARNATAKA INSTITUTE OF ENDOCRINOLOGY") is False

    def test_word_too_long_disqualifies(self):
        assert looks_like_patient_name("ABCDEFGHIJKLMNO REDDY") is False

    def test_empty_is_not_name(self):
        assert looks_like_patient_name("") is False

    def test_digits_disqualify(self):
        assert looks_like_patient_name("REDDY 1234") is False


class TestLooksLikeKierId:
    def test_standard_kier(self):
        assert looks_like_kier_id("KIER34498") is True

    def test_kier_with_space(self):
        assert looks_like_kier_id("KIER 34498") is True

    def test_kilr_ocr_variant(self):
        assert looks_like_kier_id("KILRa4498") is True

    def test_kier_6_digits(self):
        assert looks_like_kier_id("KIER175326") is True

    def test_too_few_digits(self):
        assert looks_like_kier_id("KIER123") is False

    def test_no_k_prefix(self):
        assert looks_like_kier_id("1234567") is False

    def test_only_digits_after_k(self):
        assert looks_like_kier_id("K1234567") is True

    def test_two_ocr_garbled_chars_between_prefix_and_digits(self):
        assert looks_like_kier_id("KIERIS1912") is True

    def test_empty_is_not_kier(self):
        assert looks_like_kier_id("") is False
