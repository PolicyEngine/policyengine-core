Avoid eagerly allocating empty ``Holder`` objects for every variable during ``Simulation.__init__`` and ``SimulationBuilder.build_from_dict`` when discovering ``input_variables``.
