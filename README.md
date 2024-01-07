<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- [![Issues][issues-shield]][issues-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Sentence-BERT driven by Neuro-symbolic AI
</h3>

  <p align="center">
    Neuro-symbolic based sentence matching for transformer-based encoder
    <br />
    <a href="https://github.com/Pamasi/phrase_matching_ltn/issues">Report Bug</a>
    ·
    <a href="https://github.com/Pamasi/phrase_matching_ltn/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
### About
The neuro-symbolic model used to participate in the competition "U.S. Patent Phrase to Phrase Matching"


### Built With

[![Python][Python.js]][Python-url]



<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
### Installation

1. Install python 
3. Create  a python enviroment called phrase_matching_ltn
   ```sh
   conda create phrase_matching ltn
   ```
2. Install the requirements
   ```sh
   python -m pip install -r requirements.txt

   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. Tranining
   ```sh
   python train.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [X] Train the base model
- [ ] Implemented Contrastive Learning for the base model
- [ ] Integration of neuro-symbolic AI:
    - use synonym as logical rules:
        - Forall <x,y> IsSyn(x,y) =>IsScore(x, 0.25)
        - Forall <x,y> IsContrary(x,y) =>IsScore(x, 0.0)
- [ ] Docker integration


See the [open issues](https://github.com/Pamasi/phrase_matching_ltn/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Paolo Dimasi - paolo.dimasi@outlook.com

Project Link: [https://github.com/Pamasi/phrase_matching_ltn](https://github.com/Pamasi/phrase_matching_ltn)

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Pamasi/phrase_matching_ltn.svg?style=for-the-badge
[contributors-url]: https://github.com/Pamasi/phrase_matching_ltn/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Pamasi/phrase_matching_ltn.svg?style=for-the-badge
[forks-url]: https://github.com/Pamasi/phrase_matching_ltn/network/members
[stars-shield]: https://img.shields.io/github/stars/Pamasi/phrase_matching_ltn.svg?style=for-the-badge
[stars-url]: https://github.com/Pamasi/phrase_matching_ltn/stargazers
[issues-shield]: https://img.shields.io/github/issues/Pamasi/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/Pamasi/phrase_matching_ltn/issues
[license-shield]: https://img.shields.io/github/license/Pamasi/phrase_matching_ltn.svg?style=for-the-badge
[license-url]: https://github.com/Pamasi/phrase_matching_ltn/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/paolo-dimasi


[Python-url]: https://www.Python-lang.org/
[Python.js]: https://img.shields.io/badge/Python-20232A?style=for-the-badge&logo=Python&logoColor=61DAFB


