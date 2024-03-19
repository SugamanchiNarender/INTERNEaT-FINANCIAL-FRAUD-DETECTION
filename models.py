from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class Financial_Fraud_Prediction(models.Model):

    Customer_Email= models.CharField(max_length=300)
    customerPhone= models.CharField(max_length=300)
    customerDevice= models.CharField(max_length=300)
    customerIPAddress= models.CharField(max_length=300)
    customerBillingAddress= models.CharField(max_length=300)
    No_Transactions= models.CharField(max_length=300)
    No_Orders= models.CharField(max_length=300)
    No_Payments= models.CharField(max_length=300)
    Prediction = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



