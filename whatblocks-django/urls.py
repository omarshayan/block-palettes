from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('palette/', include('palette.urls')),
    path('admin/', admin.site.urls),
]
