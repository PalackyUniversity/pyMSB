{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% set selected_methods = [] %}
    {% for item in methods %}
    {% if (not item in inherited_members) or item == 'map' %}
    {% set _ = selected_methods.append(item) %}
    {% endif %}
    {%- endfor %}

    {% block methods %}
    {% if selected_methods %}

    .. rubric:: Methods

    .. autosummary::
        :toctree: classmethods
        :template: method.rst

    {% for item in selected_methods %}
        {{objname}}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
