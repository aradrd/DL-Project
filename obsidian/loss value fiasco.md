2 changes were done:
- Reduced the number of files scanned to 4
- Used the same dataset for train and test.

after a single gradient step test gives huge numbers, but the 2nd train doesn't (???).
maybe we're not using the same weights for some reason.
try and debug the weights / the model itself.