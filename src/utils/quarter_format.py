from datetime import datetime

class QuarterFormatter:
    year: int
    quarter: int

    def __init__(self, year: int, quarter: int):
        self.year = year
        self.quarter = quarter

    @staticmethod
    def get_quarter(date_str: datetime):
        return QuarterFormatter(
            year=date_str.year,
            quarter=((date_str.month - 1) // 3) + 1
        )
    
    def __sub__(self, other: 'QuarterFormatter') -> int:
        return (self.year - other.year) * 4 + (self.quarter - other.quarter)
    
    def __str__(self) -> str:
        return f"{self.year}-Q{self.quarter}"

    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuarterFormatter):
            return False
        return self.year == other.year and self.quarter == other.quarter

    def __hash__(self) -> int:
        return hash((self.year, self.quarter))
    
    def __lt__(self, other: 'QuarterFormatter') -> bool:
        if self.year != other.year:
            return self.year < other.year
        return self.quarter < other.quarter

    def __le__(self, other: 'QuarterFormatter') -> bool:
        return self < other or self == other