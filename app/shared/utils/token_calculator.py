class TokenCostCalculator:
    """Utility to calculate estimated token usage and exact costs for Gemini 2.5 Flash."""

    INPUT_PRICE_PER_M = 0.30
    OUTPUT_PRICE_PER_M = 2.50

    ARABIC_TOKEN_FACTOR = 2.8

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count for Arabic text based on word count."""
        if not text:
            return 0
        return int(len(text.split()) * cls.ARABIC_TOKEN_FACTOR)

    @classmethod
    def calculate_cost_per_1k(cls, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of 1,000 similar queries using Gemini 2.5 Flash pricing."""
        input_cost = (input_tokens * cls.INPUT_PRICE_PER_M) / 1_000_000
        output_cost = (output_tokens * cls.OUTPUT_PRICE_PER_M) / 1_000_000

        return (input_cost + output_cost) * 1000