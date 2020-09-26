from django.db import models

# Create your models here.

class PIN(models.Model):
    PINid = models.IntegerField(primary_key=True)
    Type = models.CharField(max_length=100)
    rocation_n = models.IntegerField()
    rocation_w = models.IntegerField()
    data = models.CharField(max_length=100)
