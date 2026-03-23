:orphan:

{% set cls = class.split('.')[-1] %}
{% set title = cls + '.' + name %}
{{ title | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ fullname }}
