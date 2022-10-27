import numpy as np

def test_one_hot_encoding(one_hot_encoding):
    tx = np.array([[1.9999, 0],
                   [-999, 1],
                   [-2.3, 2],
                   [-999, 3]])
    expected_tx = np.array([[1.9999,1,0,0,0], 
                            [-999,0,1,0,1], 
                            [-2.3,0,0,1,0], 
                            [-999,0,0,0,1]])
    pr_tx = one_hot_encoding(tx, index=1, num=3)
    
    np.testing.assert_allclose(pr_tx, expected_tx)
    assert pr_tx.shape == expected_tx.shape
    print(f"✅ tests passed successfully !")
    
def test_ones_concatenate(ones_concatenate):
    tx = np.array([[1.9999, 0],
                   [-999, 1],
                   [-2.3, 2],
                   [-999, 3]])
    expected_tx = np.array([[1.9999, 0,1],
                   [-999,1,1],
                   [-2.3,2,1],
                   [-999,3,1]])
    pr_tx = ones_concatenate(tx)
    
    np.testing.assert_allclose(pr_tx, expected_tx)
    assert pr_tx.shape == expected_tx.shape
    print(f"✅ tests passed successfully !")
    
def test_handle_undefined_values(handle_undefined_values):
    tx = np.array([[1,0,1,1,1,1,1,1,1,1,1],
                   [1,1,1,1,1,1,1,1,1,1,1],
                   [2,2,1,1,1,1,1,1,1,1,1],
                   [3,3,1,1,1,1,1,1,1,1,1],
                   [3,4,1,1,1,1,1,1,1,1,1],
                   [-999,0,1,1,1,1,1,1,1,1,1]])
    expected_tx = np.array([[1,0,1,1,1,1,1,1,1,1,1],
                   [1,1,1,1,1,1,1,1,1,1,1],
                   [2,2,1,1,1,1,1,1,1,1,1],
                   [3,3,1,1,1,1,1,1,1,1,1],
                   [3,4,1,1,1,1,1,1,1,1,1],
                   [2,0,1,1,1,1,1,1,1,1,1]])
    pr_tx = handle_undefined_values(tx,total=False)
    
    np.testing.assert_allclose(pr_tx, expected_tx)
    assert pr_tx.shape == expected_tx.shape
    print(f"✅ tests passed successfully !")