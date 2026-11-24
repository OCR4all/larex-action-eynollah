# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.alias_generators import to_camel

MAX_COLUMNS = 6


@dataclass(frozen=True)
class LayoutOptions:
    lines_only: bool = False
    full_layout: bool = True
    tables: bool = False
    curved_lines: bool = False
    model_reading_order: bool = False
    right_to_left: bool = False
    marginalia: str = 'both'
    split_paragraphs_at_marginalia: bool = False
    crop_page: bool = True
    binarize: bool = False
    min_columns: int = 0
    max_columns: int = 0
    region_separation: float = 0.1
    line_separation: float = 0.1
    polygon_tolerance: float = 1.0


class RunParameters(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra='ignore', allow_inf_nan=False)

    mode: Literal['layout', 'lines'] = 'layout'
    full_layout: bool = True
    tables: bool = False
    curved_lines: bool = False
    reading_order: Literal['heuristic', 'model'] = 'heuristic'
    right_to_left: bool = False
    marginalia: Literal['both', 'left', 'right', 'off'] = 'both'
    split_paragraphs_at_marginalia: bool = False
    crop_page: bool = True
    binarize: bool = False
    min_columns: int = Field(default=0, ge=0, le=MAX_COLUMNS)
    max_columns: int = Field(default=0, ge=0, le=MAX_COLUMNS)
    region_separation: float = Field(default=0.1, ge=0, le=1)
    line_separation: float = Field(default=0.1, ge=0, le=1)
    polygon_tolerance: float = Field(default=1.0, ge=0, le=20)

    @model_validator(mode='after')
    def check_column_range(self) -> Self:
        if self.min_columns and self.max_columns and self.min_columns > self.max_columns:
            raise ValueError('minColumns must not be greater than maxColumns')
        return self

    def to_options(self) -> LayoutOptions:
        layout = self.mode == 'layout'
        constrained = bool(self.min_columns or self.max_columns)
        return LayoutOptions(
            lines_only=not layout,
            full_layout=layout and self.full_layout,
            tables=layout and self.tables,
            curved_lines=layout and self.curved_lines,
            model_reading_order=layout and self.reading_order == 'model',
            right_to_left=self.right_to_left,
            marginalia=self.marginalia if layout else 'off',
            split_paragraphs_at_marginalia=layout and self.split_paragraphs_at_marginalia,
            crop_page=self.crop_page,
            binarize=self.binarize,
            min_columns=(self.min_columns or 1) if constrained else 0,
            max_columns=(self.max_columns or MAX_COLUMNS) if constrained else 0,
            region_separation=self.region_separation,
            line_separation=self.line_separation,
            polygon_tolerance=self.polygon_tolerance,
        )


class InvalidParameters(ValueError):
    pass


def parse_parameters(raw: dict[str, Any]) -> LayoutOptions:
    try:
        return RunParameters.model_validate(raw).to_options()
    except ValidationError as error:
        details = '; '.join(
            ': '.join(filter(None, ['.'.join(map(str, issue['loc'])), issue['msg']])) for issue in error.errors()
        )
        raise InvalidParameters(f'Invalid parameters: {details}') from error
