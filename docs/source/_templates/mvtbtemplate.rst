{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods and module != 'machinevisiontoolbox.Sources' %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: {{ objname }}
      :nosignatures:

   {% for item in methods | reject('equalto', '__init__') | list | sort(case_sensitive=False) %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes and module != 'machinevisiontoolbox.Sources' %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree: {{ objname }}
      :nosignatures:

   {% for item in attributes | sort(case_sensitive=False) %}
   {% if item | lower not in (methods | reject('equalto', '__init__') | map('lower') | list) %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
