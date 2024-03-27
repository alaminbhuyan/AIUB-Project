from django.contrib import admin
from app1.models import PatientRegistration, DoctorRegistration, PatientRecords

# Register your models here.
@admin.register(PatientRegistration)
class PatientRegistrationAdmin(admin.ModelAdmin):
    list_display = ('id','user', 'phone_number' ,'gender' , 'date_of_birth')

@admin.register(DoctorRegistration)
class DoctorRegistrationAdmin(admin.ModelAdmin):
    list_display = ('id', 'doctor_user', 'phone_number', 'doctor_speciality')


@admin.register(PatientRecords)
class PatientRecordsAdmin(admin.ModelAdmin):
    list_display = ('id','patient_registration', 'name', 'age', 'gender', 
                    'heart_rate','stress_level',
                    'diabetes','chestDiseaseDuration', 
                    'blood_pressure', 'xray_img')