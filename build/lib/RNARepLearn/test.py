import gin

@gin.configurable
def test(test_var):
    print(test_var)