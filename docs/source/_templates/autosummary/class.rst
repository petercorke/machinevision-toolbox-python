{{ objname | escape | underline}}

{% if objname == 'Kernel' and module == 'machinevisiontoolbox' %}
.. currentmodule:: machinevisiontoolbox.Kernel

.. autoclass:: machinevisiontoolbox.Kernel.Kernel
{% else %}
.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{% endif %}

{% if objname == 'Kernel' and module == 'machinevisiontoolbox' %}
{% set member_toctree = 'kernel/' %}
{% elif module == 'machinevisiontoolbox' %}
{% set member_toctree = objname + '/' %}
{% else %}
{% set member_toctree = objname | lower + '/' %}
{% endif %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: {{ member_toctree }}
   {% for item in methods | reject('equalto', '__init__') | list | sort(case_sensitive=False) %}
      {% if objname == 'Kernel' and module == 'machinevisiontoolbox' -%}
      ~Kernel.{{ item }}
      {%- else -%}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Properties') }}

   .. autosummary::
      :toctree: {{ member_toctree }}
   {% for item in attributes | sort(case_sensitive=False) %}
      {% if objname == 'Kernel' and module == 'machinevisiontoolbox' -%}
      ~Kernel.{{ item }}
      {%- else -%}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

