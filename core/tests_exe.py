import test_client
import test_model
import test_maincontroller
import test_review
import unittest


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_client.TestCliente))
    suite.addTest(unittest.makeSuite(test_model.TestModel))
    suite.addTest(unittest.makeSuite(test_maincontroller.TestController))
    suite.addTest(unittest.makeSuite(test_review.TestReview))
    unittest.TextTestRunner(verbosity=2).run(suite)