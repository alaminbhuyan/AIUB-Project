from django.db import models
from django.contrib.auth.models import User
from PIL import Image
import os

# Create your models here.
class PatientRegistration(models.Model):
    user = models.OneToOneField(to=User, on_delete=models.CASCADE, related_name='profile')
    phone_number = models.CharField(max_length=15, null=False, blank=False)
    gender = models.CharField(max_length=20, null=False, blank=False)
    date_of_birth = models.DateField()

class PatientRecords(models.Model):
    patient_registration = models.ForeignKey(PatientRegistration, on_delete=models.CASCADE, related_name='records')
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    phone_number = models.CharField(max_length=20)
    blood_pressure = models.CharField(max_length=20)
    heart_rate = models.CharField(max_length=20)
    stress_level = models.CharField(max_length=20)
    diabetes = models.CharField(max_length=20)
    chestDiseaseDuration = models.CharField(max_length=30)
    healthIssues = models.TextField()
    lastCheckupDate = models.DateField()
    previousDoctorAdvice = models.TextField()
    recentCondition = models.TextField()
    xray_img = models.ImageField(verbose_name="X_Ray_Img", upload_to='xray_img', null=False, blank=False)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        img = Image.open(self.xray_img.path)
        print(self.xray_img.path)
        # Extract the directory and filename from the path
        directory, filename = os.path.split(self.xray_img.path)
        # Extract the extension from the filename
        base_name, extension = os.path.splitext(filename)
        # Define the new filename
        new_filename = 'CHNCXR_0001_0' + extension
        # Construct the new path
        new_path = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(self.xray_img.path, new_path)


        if img.height > 300 or img.width > 300:
            output_size = (210, 210)
            img.thumbnail(size=output_size)
            img.save(self.xray_img.path)



class DoctorRegistration(models.Model):
    doctor_user = models.OneToOneField(to=User, on_delete=models.CASCADE, related_name='doctor_profile')
    phone_number = models.CharField(max_length=15, null=False, blank=False)
    doctor_speciality = models.CharField(max_length=50, null=False, blank=False)
