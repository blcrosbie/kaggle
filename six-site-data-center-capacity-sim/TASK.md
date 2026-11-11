# TASK — Six-Site Data Center Capacity Simulator

> **Source:** Upwork job posting, captured 2026-11-09 ("Six-Site Data Center Capacity Simulator", posted worldwide, hourly $20–$40, intermediate, < 1 month).
> **Personal context:** Built as a self-directed portfolio project regardless of whether the contract is awarded. The plan is to meet the full SOW below, then go beyond it with a Monte Carlo layer (stochastic growth and expansion delays, giving probability-of-exhaustion ranges instead of single dates), drawing on a sustainability-systems background.
>
> The SOW is reproduced verbatim below.

---

# Statement of Work (SOW)
## Six-Site Data Center Capacity Planning & Executive Decision Simulator

**Project Type:** Python / Streamlit Web Application Development  
**Engagement:** Fixed-price, milestone-based  
**Estimated Duration:** 4–6 weeks  
**Deployment:** Streamlit Community Cloud  
**Data Classification:** Synthetic / Demonstration Only  
**Version:** 1.0 — November 2026

---

## 1. Project Overview

Seeking an experienced Python and Streamlit developer to design, build, test, and deploy an interactive **Six-Site Data Center Capacity Planning Simulator**.

The application will demonstrate how regional data center operations leaders evaluate infrastructure capacity, forecast growth, identify operational risks, and prioritize capital investments across a portfolio of six facilities.

The intended audience includes data center directors, regional operations managers, infrastructure executives, and technical hiring managers.

The application must provide a polished, executive-ready user experience suitable for inclusion in a professional LinkedIn portfolio.

**The application is a demonstration and decision-support tool, not a production data center management system.** All facility information and operational data must be synthetic.

## 2. Project Objectives

The completed application must:

1. Consolidate six data centers into a single executive portfolio view.
2. Model electrical power, cooling, and rack-space capacity.
3. Forecast infrastructure demand over 6–36 months.
4. Identify projected capacity constraints and exhaustion dates.
5. Support interactive scenario planning and what-if analysis.
6. Estimate infrastructure expansion requirements and capital investment costs.
7. Prioritize sites based on capacity risk and investment urgency.
8. Produce executive-level recommendations supported by transparent calculations.
9. Allow users to download scenario results.
10. Be publicly accessible through a working Streamlit deployment.

## 3. Technology Requirements

**Required:**
- Python 3.11 or later
- Streamlit
- Pandas and NumPy
- Plotly for interactive visualizations
- CSV-based synthetic data
- GitHub for version control and source-code delivery
- Streamlit Community Cloud for deployment

The developer may propose additional open-source libraries with prior approval.

The application should not require paid APIs, commercial databases, or recurring infrastructure expenses for the demonstration workload.

## 4. Functional Requirements

### Module 1 — Executive Portfolio Overview

Create a dashboard displaying the aggregate performance of six fictional data centers.

**Required KPIs:**
- Total available electrical capacity (MW)
- Current electrical demand (MW)
- Electrical utilization (%)
- Remaining electrical headroom (MW)
- Total cooling capacity and utilization
- Total rack capacity and occupied racks
- Forecasted demand at 12, 24, and 36 months
- Number of facilities approaching capacity constraints
- Earliest projected site capacity exhaustion
- Portfolio-level capacity risk

**Visualizations:**
- Portfolio demand versus capacity forecast
- Power utilization chart
- Six-site comparison table
- Capacity risk indicators
- Executive insights and recommended actions

All metrics must recalculate when simulation parameters change.

### Module 2 — Facility-Level Capacity Analysis

Each of the six facilities must have configurable synthetic data covering:

| Data Field | Requirement |
|---|---|
| Site ID | DC1–DC6 |
| Site Name | Fictional location or facility name |
| Electrical Capacity | MW |
| Current Electrical Demand | MW |
| Cooling Capacity | Defined and documented units |
| Current Cooling Load | Same units as cooling capacity |
| Total Rack Capacity | Number of racks |
| Occupied Racks | Number of racks |
| Annual Demand Growth | Percentage |
| Planned Expansion | Capacity, cost, and activation month |
| Risk Status | Low / Medium / High / Critical |

Users must be able to select individual facilities and compare their utilization, forecasts, constraints, and expansion requirements.

### Module 3 — Capacity Forecasting

Develop a forecasting engine covering 6, 12, 18, 24, 30, and 36 months, with monthly calculations.

**Requirements:**
- Compound annual demand growth
- Site-specific growth assumptions
- Monthly capacity and demand projections
- Estimated capacity exhaustion dates
- Configurable utilization thresholds
- Planned expansion activation dates
- Power, cooling, and rack-space constraints
- Identification of the first limiting infrastructure resource

Forecasting assumptions must be documented, and users must be able to distinguish actual synthetic baseline data from projected results.

### Module 4 — Interactive Scenario Planning

Provide sidebar controls and predefined scenario buttons.

**User-adjustable inputs:**
- Forecast horizon
- Portfolio-wide annual growth
- Site-specific annual growth
- Power expansion (MW)
- Cooling expansion
- Additional rack capacity
- Expansion activation month
- Operational efficiency improvement (%)
- Capital investment budget

**Predefined scenarios:**
1. Baseline — Current operating assumptions
2. High Growth — 25% annual demand growth
3. Infrastructure Expansion — Add capacity to a selected facility
4. Efficiency Optimization — Reduce modeled demand by 10%
5. Budget-Constrained — Prioritize projects within a specified capital budget

Changes must update dashboard KPIs, charts, site risks, and recommendations without manual data reloads.

Include a Reset to Baseline control.

### Module 5 — Capital Investment Planning

Create a simplified capital planning model.

**Requirements:**
- Configurable cost per additional MW
- Configurable cooling expansion cost
- Configurable cost per additional rack
- Estimated total project cost
- Capital budget versus proposed investment
- Project activation timing
- Estimated capacity gained
- Capacity headroom after investment
- Prioritized investment opportunities

Provide a transparent ranking methodology based on capacity urgency, risk, and estimated cost.

The tool must not claim precise engineering estimates or financial ROI without sufficient assumptions.

### Module 6 — Executive Decision Support

Develop an executive recommendations panel that identifies:

- Facilities at risk of reaching capacity limits
- Recommended expansion timing
- Sites requiring immediate attention
- Opportunities for capacity optimization
- Investments exceeding the available budget
- Potential consequences of deferring infrastructure expansion

Each recommendation must reference the underlying modeled metrics and assumptions.

Use deterministic, rules-based logic. AI-generated recommendations are not required for the initial release.

### Module 7 — Reporting and Export

Users must be able to download:

- Facility-level capacity data (CSV)
- Monthly capacity forecast (CSV)
- Scenario assumptions and results (CSV)
- Capital investment comparison (CSV)

A printable executive summary is desirable but optional.

## 5. User Interface and Design

The application should have a modern, professional dashboard design inspired by enterprise infrastructure monitoring platforms.

**Preferred visual style:**
- Dark navy theme
- Blue, green, amber, and red status indicators
- Executive KPI cards
- Interactive Plotly charts
- Clear navigation sidebar
- Responsive layout for desktop and laptop browsers
- Consistent typography and spacing

**Navigation pages:**
1. Executive Overview
2. Capacity Forecasting
3. Scenario Planning
4. Executive Decisions

A reference dashboard mockup will be supplied to the selected developer. The final application should closely follow its visual structure while maintaining usability and accessibility.

## 6. Data and Calculation Requirements

The developer must generate a reproducible synthetic dataset representing six fictional data centers.

The dataset must contain realistic but invented values, with no employer-specific, classified, proprietary, or sensitive information.

**Calculation requirements:**
- Utilization = Current Demand / Available Capacity
- Headroom = Available Capacity − Current Demand
- Forecast Demand = Baseline Demand × (1 + Annual Growth Rate) ^ (Months / 12)
- Exhaustion = First forecast period when demand exceeds available capacity
- Portfolio Utilization = Sum of Demand / Sum of Capacity

The developer must also implement and document appropriate calculations for cooling, rack occupancy, expansion costs, and site risk.

Portfolio-level exhaustion must not hide earlier site-level constraints. The executive dashboard must prominently display the earliest facility/resource constraint.

All assumptions, units, formulas, thresholds, and limitations must be documented.

## 7. Nonfunctional Requirements

- Application must deploy successfully on Streamlit Community Cloud.
- No login should be required to access the public demonstration.
- No proprietary information or real infrastructure credentials may be included.
- Application must provide clear error messages for invalid inputs.
- Calculations must remain consistent across all dashboard pages.
- Application must operate without a paid backend service.
- Code must be modular, maintainable, and reasonably documented.
- Interactive changes should complete promptly under normal demonstration conditions.
- Application must display a visible **Synthetic Data — Demonstration Only** notice.

## 8. Deliverables

The developer must provide:

1. Complete Streamlit application source code.
2. `requirements.txt` with dependencies.
3. Six-site synthetic dataset and generation script.
4. Capacity forecasting and scenario calculation modules.
5. Interactive dashboard with all four pages.
6. Capital investment planning module.
7. Export functionality.
8. GitHub repository containing the complete project.
9. Deployment to Streamlit Community Cloud.
10. README with installation, configuration, and deployment instructions.
11. Data dictionary and calculation methodology.
12. Automated tests covering core calculations and scenario behavior.
13. Recorded walkthrough or live knowledge-transfer session.
14. Resolution of acceptance-test defects before final handover.

All source code and deliverables must be transferred to the client upon final payment, with ownership terms established in the Upwork contract.

## 9. Project Milestones and Acceptance Criteria

| Milestone | Deliverable | Payment |
|---|---|---:|
| 1 | Architecture, synthetic dataset, UI design, calculation specifications | 15% |
| 2 | Executive Overview and facility comparison | 25% |
| 3 | Forecasting engine and scenario planning | 25% |
| 4 | Capital planning and executive recommendations | 20% |
| 5 | Testing, documentation, deployment, and handover | 15% |

**Final acceptance requires:**
- All six facilities represented correctly.
- All four pages functional.
- Changes to inputs reflected consistently across applicable metrics.
- Forecast calculations verified against documented test cases.
- Earliest site-level capacity constraints correctly identified.
- No negative physical capacity inputs or logically inconsistent results accepted without validation.
- All CSV exports working.
- Public Streamlit deployment operational.
- Complete source code delivered.
- No unresolved critical or high-severity defects.

## 10. Developer Qualifications

Applicants should demonstrate:

- Strong Python development experience
- Proven Streamlit applications
- Experience with Pandas and Plotly
- Data modeling and forecasting skills
- Experience building executive dashboards
- GitHub and cloud deployment experience
- Ability to document calculations and assumptions

Experience with data centers, infrastructure capacity planning, energy management, or operational analytics is highly desirable.

## 11. Proposal Submission Requirements

Please include:

1. Links or screenshots of at least two Streamlit applications you have built.
2. Your proposed architecture and implementation approach.
3. Estimated fixed price and delivery timeline.
4. Relevant experience with forecasting or infrastructure analytics.
5. Any technical assumptions or limitations.
6. Confirmation that you can deploy and transfer the completed application.
7. Your proposed testing and handover approach.

Generic proposals without relevant Streamlit examples may not be considered.

## 12. Out of Scope

The initial release does not require:

- Integration with live production data centers
- DCIM, BMS, CMDB, or ServiceNow integrations
- Real-time telemetry ingestion
- Production-grade authentication or access control
- AI/LLM integrations
- Production financial forecasting
- Detailed electrical or mechanical engineering design
- Power BI integration
- Mobile-native application development

These capabilities may be considered in a future phase.

## 13. Confidentiality and Ownership

The application must use synthetic demonstration data only.

The developer must not incorporate proprietary employer information, confidential infrastructure configurations, or restricted materials.

All custom source code, datasets, documentation, and other project deliverables must be provided to the client under the agreed ownership terms.

Any third-party packages, templates, or assets must have appropriate licenses documented before use.

---

**Project Success Definition**

A successful project will deliver a publicly accessible, visually compelling, and technically credible Streamlit application that demonstrates how a regional data center operations leader can evaluate six facilities, forecast capacity requirements, identify operational risks, and make defensible infrastructure investment decisions.

---

## Posting metadata

- Posted: 2 hours before capture (2026-11-09)
- Location: Worldwide
- Rate: Hourly, $20.00 – $40.00
- Duration: Less than 1 month
- Experience level: Intermediate ("a mix of experience and value")
