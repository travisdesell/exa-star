class InnovationGenerator:
    innovation_counter: int = 0

    @staticmethod
    def get_innovation_number() -> int:
        """
        Returns:
            The next unique innovation number.
        """
        number = InnovationGenerator.innovation_counter
        InnovationGenerator.innovation_counter += 1
        return number
