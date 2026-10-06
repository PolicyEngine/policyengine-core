Looking up parameters with empty key arrays now returns an empty array at every level of a chained lookup, instead of raising `IndexError` once a level holds numeric values.
