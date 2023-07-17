from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

from pseudopeople.constants.metadata import DATEFORMATS, DatasetNames
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES


class DtypeNames:
    """Container of expected dtype names"""

    CATEGORICAL = "category"
    DATETIME = "datetime64[ns]"
    OBJECT = "object"


@dataclass
class Column:
    name: str
    noise_types: Tuple[ColumnNoiseType, ...] = tuple()
    dtype_name: str = DtypeNames.OBJECT  # string dtype is 'object'


class __Columns(NamedTuple):
    """Container that contains information about columns that have potential to be noised"""

    age: Column = Column(
        "age",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.copy_from_household_member,
            NOISE_TYPES.misreport_age,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    city: Column = Column(
        "city",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    dob: Column = Column(
        "date_of_birth",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.copy_from_household_member,
            NOISE_TYPES.swap_month_and_day,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    employer_city: Column = Column(
        "employer_city",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    employer_id: Column = Column(
        "employer_id",
    )
    employer_name: Column = Column(
        "employer_name",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    employer_state: Column = Column(
        "employer_state",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    employer_street_name: Column = Column(
        "employer_street_name",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    employer_street_number: Column = Column(
        "employer_street_number",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    employer_unit_number: Column = Column(
        "employer_unit_number",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    employer_zipcode: Column = Column(
        "employer_zipcode",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_zipcode_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    first_name: Column = Column(
        "first_name",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.use_nickname,
            NOISE_TYPES.use_fake_name,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    guardian_id: Column = Column(
        "guardian_id",
    )
    household_id: Column = Column(
        "household_id",
    )
    income: Column = Column(
        "income",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    itin: Column = Column(
        "itin",
        (
            NOISE_TYPES.leave_blank,
            # NOISE_TYPES.copy_from_household_member,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    last_name: Column = Column(
        "last_name",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.use_fake_name,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    mailing_city: Column = Column(
        "mailing_address_city",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    mailing_po_box: Column = Column(
        "mailing_address_po_box",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    mailing_state: Column = Column(
        "mailing_address_state",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    mailing_street_name: Column = Column(
        "mailing_address_street_name",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    mailing_street_number: Column = Column(
        "mailing_address_street_number",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    mailing_unit_number: Column = Column(
        "mailing_address_unit_number",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    mailing_zipcode: Column = Column(
        "mailing_address_zipcode",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_zipcode_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    middle_initial: Column = Column(
        "middle_initial",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    race_ethnicity: Column = Column(
        "race_ethnicity",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    relation_to_reference_person: Column = Column(
        "relation_to_reference_person",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    sex: Column = Column(
        "sex",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    simulant_id: Column = Column(
        "simulant_id",
    )
    spouse_household_id: Column = Column(
        "spouse_household_id",
    )
    ssa_event_date: Column = Column(
        "event_date",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.swap_month_and_day,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    ssa_event_type: Column = Column(
        "event_type",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    ssn: Column = Column(
        "ssn",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.copy_from_household_member,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    state: Column = Column(
        "state",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    street_name: Column = Column(
        "street_name",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.make_phonetic_errors,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    street_number: Column = Column(
        "street_number",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    survey_date: Column = Column(
        "survey_date",
        dtype_name=DtypeNames.DATETIME,
    )
    tax_form: Column = Column(
        "tax_form",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.choose_wrong_option,
        ),
        DtypeNames.CATEGORICAL,
    )
    tax_year: Column = Column(
        "tax_year",
    )
    unit_number: Column = Column(
        "unit_number",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )
    year: Column = Column(
        "year",
    )
    zipcode: Column = Column(
        "zipcode",
        (
            NOISE_TYPES.leave_blank,
            NOISE_TYPES.write_wrong_zipcode_digits,
            NOISE_TYPES.make_ocr_errors,
            NOISE_TYPES.make_typos,
        ),
    )

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def get_column(name: str) -> Column:
        """Return the respective Column object given the column name"""
        return [c for c in COLUMNS if c.name == name][0]


COLUMNS = __Columns()


@dataclass
class Dataset:
    name: str
    columns: Tuple[Column, ...]  # This defines the output column order
    date_column_name: str
    date_format: str
    state_column_name: Optional[str]
    row_noise_types: Tuple[RowNoiseType, ...]


class __Datasets(NamedTuple):
    """NamedTuple that contains information about datasets and their related columns"""

    census: Dataset = Dataset(
        DatasetNames.CENSUS,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.household_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.relation_to_reference_person,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
            COLUMNS.year,
        ),
        date_column_name=COLUMNS.year.name,
        state_column_name=COLUMNS.state.name,
        row_noise_types=(
            NOISE_TYPES.do_not_respond,
            # NOISE_TYPES.duplication,
        ),
        date_format=DATEFORMATS.MM_DD_YYYY,
    )
    acs: Dataset = Dataset(
        DatasetNames.ACS,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.household_id,
            COLUMNS.survey_date,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
        ),
        date_column_name=COLUMNS.survey_date.name,
        state_column_name=COLUMNS.state.name,
        row_noise_types=(
            NOISE_TYPES.do_not_respond,
            # NOISE_TYPES.duplication,
        ),
        date_format=DATEFORMATS.MM_DD_YYYY,
    )
    cps: Dataset = Dataset(
        DatasetNames.CPS,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.household_id,
            COLUMNS.survey_date,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
        ),
        date_column_name=COLUMNS.survey_date.name,
        state_column_name=COLUMNS.state.name,
        row_noise_types=(
            NOISE_TYPES.do_not_respond,
            # NOISE_TYPES.duplication,
        ),
        date_format=DATEFORMATS.MM_DD_YYYY,
    )
    wic: Dataset = Dataset(
        DatasetNames.WIC,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.household_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
            COLUMNS.year,
        ),
        date_column_name=COLUMNS.year.name,
        state_column_name=COLUMNS.state.name,
        row_noise_types=(
            NOISE_TYPES.omit_row,
            # NOISE_TYPES.duplication,
        ),
        date_format=DATEFORMATS.MMDDYYYY,
    )
    ssa: Dataset = Dataset(
        DatasetNames.SSA,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.dob,
            COLUMNS.ssn,
            COLUMNS.ssa_event_type,
            COLUMNS.ssa_event_date,
        ),
        date_column_name=COLUMNS.ssa_event_date.name,
        state_column_name=None,
        row_noise_types=(
            NOISE_TYPES.omit_row,
            # NOISE_TYPES.duplication,
        ),
        date_format=DATEFORMATS.YYYYMMDD,
    )
    tax_w2_1099: Dataset = Dataset(
        DatasetNames.TAXES_W2_1099,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.household_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.mailing_street_number,
            COLUMNS.mailing_street_name,
            COLUMNS.mailing_unit_number,
            COLUMNS.mailing_po_box,
            COLUMNS.mailing_city,
            COLUMNS.mailing_state,
            COLUMNS.mailing_zipcode,
            COLUMNS.ssn,
            COLUMNS.income,
            COLUMNS.employer_id,
            COLUMNS.employer_name,
            COLUMNS.employer_street_number,
            COLUMNS.employer_street_name,
            COLUMNS.employer_unit_number,
            COLUMNS.employer_city,
            COLUMNS.employer_state,
            COLUMNS.employer_zipcode,
            COLUMNS.tax_form,
            COLUMNS.tax_year,
        ),
        date_column_name=COLUMNS.tax_year.name,
        state_column_name=COLUMNS.mailing_state.name,
        row_noise_types=(
            NOISE_TYPES.omit_row,
            # NOISE_TYPES.duplication,
        ),
        date_format=DATEFORMATS.MM_DD_YYYY,
    )
    # tax_1040: Dataset = Dataset(
    #     DatasetNames.TAXES_1040,
    #     columns=list(),
    #     date_column_name=COLUMNS.tax_year.name,
    #     state_column_name=COLUMNS.mailing_state.name,
    #     row_noise_types=(
    #         NOISE_TYPES.omit_row,
    #         # NOISE_TYPES.duplication,
    #     ),
    #     date_format=DATEFORMATS.MM_DD_YYYY,
    # )

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def get_dataset(name: str) -> Dataset:
        """Return the respective Dataset object given the dataset name"""
        return [d for d in DATASETS if d.name == name][0]


DATASETS = __Datasets()
