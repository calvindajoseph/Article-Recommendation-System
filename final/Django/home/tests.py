from django.test import TestCase, SimpleTestCase

# Create your tests here.

# testing index
class SimpleTest(SimpleTestCase):
    def test_home_page_status(self):
        response = sel.client.get('/')
        self.assertEquals(response.status_code,200)
        # if true return 200 otherwise return 404
        
