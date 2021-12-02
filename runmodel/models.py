from django.db import models

class HandWriting(models.Model):
  id = models.BigAutoField(primary_key=True, null=False)
  user_id = models.IntegerField(null=True)
  image = models.ImageField(upload_to="hand_writing/", blank=True, null=True)
