:orphan:

{% set title = class + '.' + name %}
{{ title | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ fullname }}