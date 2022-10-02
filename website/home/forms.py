from django import forms

class DateTimeRangeForm(forms.Form):
    date = forms.DateField(label='from')
    from_time = forms.TimeField(label='from_time')
    to_time = forms.TimeField(label='to_time')
