# Generated by Django 5.0 on 2023-12-20 08:39

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0003_doctorregistration_doctor_speciality'),
    ]

    operations = [
        migrations.CreateModel(
            name='PatientRecords',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('age', models.IntegerField()),
                ('gender', models.CharField(max_length=20)),
                ('phone_number', models.CharField(max_length=20)),
                ('blood_pressure', models.CharField(max_length=20)),
                ('heart_rate', models.CharField(max_length=20)),
                ('stress_level', models.CharField(max_length=20)),
                ('diabetes', models.CharField(max_length=20)),
                ('chestDiseaseDuration', models.CharField(max_length=30)),
                ('healthIssues', models.TextField()),
                ('lastCheckupDate', models.DateField()),
                ('previousDoctorAdvice', models.TextField()),
                ('recentCondition', models.TextField()),
                ('xray_img', models.ImageField(upload_to='xray_img', verbose_name='X_Ray_Img')),
                ('patient_registration', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='records', to='app1.patientregistration')),
            ],
        ),
    ]